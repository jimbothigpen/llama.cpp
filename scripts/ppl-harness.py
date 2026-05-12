#!/usr/bin/env python3
"""PPL regression harness for llama-yggdrasil.

Runs llama-perplexity against the ROCm and/or Vulkan yggdrasil installs,
writes structured JSON baselines to yggdrasil-context/ppl-baselines/, and
emits a PASS/WARN/FAIL parity verdict between backends.

Usage:
    ./scripts/ppl-harness.py --model /path/to/model.gguf --chunks 20
    ./scripts/ppl-harness.py --model /path/to/model.gguf --backends rocm
    ./scripts/ppl-harness.py --model /path/to/model.gguf --chunks 2 --no-record

Exit codes:
    0  PASS (parity gate passed, or single backend requested)
    1  FAIL (parity delta >= 1.0%)
    2  A backend failed (subprocess error or PPL parse failure)
    3  Invocation error (bad args, missing inputs, install missing)
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

HARNESS_VERSION = "0.1.0"

CONTEXT_ROOT = Path(
    "/mnt/cephfs/0/Container/systems/ai00/users/builduser/yggdrasil-context"
)
DEFAULT_WIKITEXT = CONTEXT_ROOT / "corpora" / "wikitext-2-raw" / "wiki.test.raw"
DEFAULT_BASELINE_ROOT = CONTEXT_ROOT / "ppl-baselines"
SHA256_CACHE_PATH = Path.home() / ".cache" / "yggdrasil" / "model-sha256.json"

REPO_ROOT = Path(__file__).resolve().parent.parent

PPL_LINE_RE = re.compile(
    r"^\s*Final\s+estimate:\s+PPL\s*=\s*([0-9]+\.[0-9]+)\s*(?:\+/?-\s*([0-9]+\.[0-9]+))?",
    re.IGNORECASE | re.MULTILINE,
)
TIMING_LOAD_RE = re.compile(
    r"llama_print_timings:\s+load time\s*=\s*([0-9]+\.[0-9]+)\s*ms",
    re.IGNORECASE | re.MULTILINE,
)
TIMING_EVAL_RE = re.compile(
    r"llama_print_timings:\s+eval time\s*=\s*([0-9]+\.[0-9]+)\s*ms",
    re.IGNORECASE | re.MULTILINE,
)
TOKENS_SCORED_RE = re.compile(
    r"([0-9]+)\s+tokens",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# sha256 cache
# ---------------------------------------------------------------------------

def _load_sha256_cache() -> dict:
    if SHA256_CACHE_PATH.exists():
        try:
            return json.loads(SHA256_CACHE_PATH.read_text())
        except Exception:
            pass
    return {"version": 1, "entries": {}}


def _save_sha256_cache(cache: dict) -> None:
    SHA256_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHA256_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def get_model_sha256(path: Path) -> str:
    cache = _load_sha256_cache()
    abs_path = str(path.resolve())
    st = path.stat()
    entry = cache["entries"].get(abs_path)
    if (
        entry
        and entry["size_bytes"] == st.st_size
        and entry["mtime_unix"] == int(st.st_mtime)
    ):
        return entry["sha256"]
    print(f"  computing model sha256 (caching for future runs)...", flush=True)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    sha = digest.hexdigest()
    cache["entries"][abs_path] = {
        "size_bytes": st.st_size,
        "mtime_unix": int(st.st_mtime),
        "sha256": sha,
    }
    _save_sha256_cache(cache)
    return sha


def get_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# git info
# ---------------------------------------------------------------------------

def get_git_info() -> dict:
    def run(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        return r.stdout.strip() if r.returncode == 0 else None

    commit = run(["git", "rev-parse", "--short", "HEAD"])
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty_out = run(["git", "status", "--porcelain"])
    dirty = bool(dirty_out)
    return {"commit": commit, "branch": branch, "dirty": dirty}


# ---------------------------------------------------------------------------
# wikitext bootstrap
# ---------------------------------------------------------------------------

def bootstrap_wikitext(path: Path, *, allow_fetch: bool) -> None:
    if path.exists():
        return
    if not allow_fetch:
        raise SystemExit(
            f"error: wikitext missing at {path} and --wikitext-no-fetch was set"
        )
    if path.name != "wiki.test.raw" or path.parent.name != "wikitext-2-raw":
        raise SystemExit(
            f"error: wikitext path {path} doesn't match the canonical "
            f".../wikitext-2-raw/wiki.test.raw layout; "
            f"place the file manually or pass a canonical --wikitext path"
        )
    target_dir = path.parent.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    script = REPO_ROOT / "scripts" / "get-wikitext-2.sh"
    print(f"  fetching wikitext-2 corpus via {script.name}...", flush=True)
    result = subprocess.run([str(script)], cwd=target_dir)
    if result.returncode != 0:
        raise SystemExit(f"error: get-wikitext-2.sh failed (exit {result.returncode})")
    if not path.exists():
        raise SystemExit(f"error: get-wikitext-2.sh ran but {path} still missing")


# ---------------------------------------------------------------------------
# GPU exclusivity guard
# ---------------------------------------------------------------------------

def check_gpu_idle() -> None:
    out = subprocess.run(
        ["pgrep", "-fa", "llama-perplexity"],
        capture_output=True,
        text=True,
    )
    matches = [
        line
        for line in out.stdout.splitlines()
        if "/llama-perplexity" in line
    ]
    if matches:
        raise SystemExit(
            "error: another llama-perplexity is running:\n  "
            + "\n  ".join(matches)
            + "\nresolve before re-running, or pass --no-gpu-check"
        )


# ---------------------------------------------------------------------------
# output parsing
# ---------------------------------------------------------------------------

def parse_perplexity_output(text: str) -> dict:
    result = {}
    m = PPL_LINE_RE.search(text)
    if m:
        result["ppl"] = float(m.group(1))
        result["ppl_stderr"] = float(m.group(2)) if m.group(2) else None

    m = TIMING_LOAD_RE.search(text)
    if m:
        result["load_sec"] = round(float(m.group(1)) / 1000.0, 2)

    m = TIMING_EVAL_RE.search(text)
    if m:
        result["eval_sec"] = round(float(m.group(1)) / 1000.0, 2)

    return result


# ---------------------------------------------------------------------------
# backend dispatch
# ---------------------------------------------------------------------------

def run_backend(backend: str, args: argparse.Namespace, log_dir: Path) -> dict:
    binary = Path(f"/opt/llama-yggdrasil-{backend}/bin/llama-perplexity")
    if not binary.exists():
        raise SystemExit(
            f"error: {binary} not found — is yggdrasil installed for {backend}?\n"
            f"  (try: sudo ninja install in build-{backend}-$(hostname)/)"
        )

    log_path = log_dir / f"ppl-{backend}.log"
    cmd = [
        str(binary),
        "-m", str(args.model),
        "-f", str(args.wikitext),
        "--no-mmap",
        "-fa", "on",
        "-ngl", str(args.ngl),
        "-c", str(args.ctx_size),
    ]
    if args.chunks > 0:
        cmd.extend(["--chunks", str(args.chunks)])
    if args.extra_flag:
        cmd.extend(args.extra_flag)

    if not args.no_gpu_check:
        check_gpu_idle()

    print(f"[{backend}] {' '.join(cmd)}", flush=True)
    start = time.monotonic()
    if args.verbose:
        with log_path.open("wb") as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            for line in proc.stdout:
                sys.stderr.buffer.write(line)
                logf.write(line)
            rc = proc.wait()
    else:
        with log_path.open("wb") as logf:
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
            rc = proc.wait()
    wallclock = time.monotonic() - start

    log_text = log_path.read_text(errors="replace")
    parsed = parse_perplexity_output(log_text)

    n_tokens_scored = parsed.get("n_tokens_scored")
    if n_tokens_scored is None and args.chunks > 0:
        n_tokens_scored = args.chunks * args.ctx_size

    return {
        "backend": backend,
        "binary": str(binary),
        "cmd": cmd,
        "params": {
            "chunks": args.chunks,
            "ctx_size": args.ctx_size,
            "ngl": args.ngl,
            "extra_flags": args.extra_flag or [],
        },
        "result": {
            "ppl": parsed.get("ppl"),
            "ppl_stderr": parsed.get("ppl_stderr"),
            "n_tokens_scored": n_tokens_scored,
            "exit_code": rc,
        },
        "timing": {
            "wallclock_sec": round(wallclock, 1),
            "load_sec": parsed.get("load_sec"),
            "eval_sec": parsed.get("eval_sec"),
        },
        "log_path": str(log_path),
    }


# ---------------------------------------------------------------------------
# parity gate
# ---------------------------------------------------------------------------

def parity_verdict(ppl_a: float, ppl_b: float) -> tuple:
    if min(ppl_a, ppl_b) <= 0:
        return ("ERROR", float("nan"))
    delta_pct = abs(ppl_a - ppl_b) / min(ppl_a, ppl_b) * 100.0
    if delta_pct < 0.5:
        return ("PASS", delta_pct)
    elif delta_pct < 1.0:
        return ("WARN", delta_pct)
    else:
        return ("FAIL", delta_pct)


# ---------------------------------------------------------------------------
# baseline I/O
# ---------------------------------------------------------------------------

def baseline_path(
    baseline_root: Path, model_basename: str, git_info: dict, host: str, backend: str, chunks: int
) -> Path:
    commit = git_info.get("commit") or "unknown"
    if git_info.get("dirty"):
        commit = f"dirty-{commit}"
    filename = f"{host}__{backend}__{chunks}chunks.json"
    return baseline_root / model_basename / commit / filename


def write_baseline(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))


def ensure_baseline_readme(baseline_root: Path) -> None:
    readme = baseline_root / "README.md"
    if readme.exists():
        return
    baseline_root.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    readme.write_text(
        f"# ppl-baselines\n\n"
        f"Generated by `scripts/ppl-harness.py`. Created {now}.\n\n"
        f"Layout: `<model-basename>/<git-commit>/<host>__<backend>__<chunks>chunks.json`\n\n"
        f"Each JSON file is one llama-perplexity run. "
        f"Future tooling reads these to detect regressions.\n"
    )


# ---------------------------------------------------------------------------
# first-run init
# ---------------------------------------------------------------------------

def ensure_dirs(baseline_root: Path) -> None:
    SHA256_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    baseline_root.mkdir(parents=True, exist_ok=True)
    ensure_baseline_readme(baseline_root)


# ---------------------------------------------------------------------------
# summary printing
# ---------------------------------------------------------------------------

def print_summary(
    results: list,
    model_path: Path,
    wikitext_path: Path,
    git_info: dict,
    host: str,
    chunks: int,
    baseline_paths: list,
    verdict_str: str,
    model_sha: str,
    wikitext_sha: str,
) -> None:
    model_sha_short = model_sha[:8] if model_sha else "n/a"
    wikitext_sha_short = wikitext_sha[:8] if wikitext_sha else "n/a"
    commit = git_info.get("commit", "unknown")
    if git_info.get("dirty"):
        commit = f"dirty-{commit}"
    branch = git_info.get("branch", "")

    print()
    print("=== PPL Harness Summary ===")
    print(f"Model:    {model_path.name}  (sha256 {model_sha_short})")
    print(f"Wikitext: {wikitext_path.name}  (sha256 {wikitext_sha_short})")
    print(f"Commit:   {commit} ({branch})")
    print(f"Host:     {host}")
    print(f"Chunks:   {chunks if chunks > 0 else 'all'}")
    print()

    for r in results:
        backend = r["backend"]
        ppl = r["result"]["ppl"]
        ppl_se = r["result"]["ppl_stderr"]
        wall = r["timing"]["wallclock_sec"]
        if ppl is not None:
            se_str = f" ± {ppl_se:.4f}" if ppl_se is not None else ""
            print(f"  {backend:<8}  PPL = {ppl:.4f}{se_str}   ({wall}s)")
        else:
            rc = r["result"]["exit_code"]
            print(f"  {backend:<8}  FAILED (exit {rc}) — see {r['log_path']}")

    print()
    print(f"Parity:   {verdict_str}")

    if baseline_paths:
        print()
        print("Baselines written:")
        for p in baseline_paths:
            print(f"  {p}")


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, type=Path, metavar="PATH",
        help="path to a .gguf model file",
    )
    parser.add_argument(
        "--backends", default="rocm,vulkan", metavar="LIST",
        help="comma-separated backends to run: rocm,vulkan (default: rocm,vulkan)",
    )
    parser.add_argument(
        "--wikitext", type=Path, default=DEFAULT_WIKITEXT, metavar="PATH",
        help=f"wikitext-2 test corpus (default: {DEFAULT_WIKITEXT})",
    )
    parser.add_argument(
        "--chunks", type=int, default=0, metavar="N",
        help="max chunks to score; 0 = whole corpus (default: 0)",
    )
    parser.add_argument(
        "--ctx-size", type=int, default=512, metavar="N",
        help="context size passed as -c (default: 512)",
    )
    parser.add_argument(
        "--ngl", type=int, default=99, metavar="N",
        help="GPU layers -ngl (default: 99; use 0 for CPU-only)",
    )
    parser.add_argument(
        "--extra-flag", action="append", metavar="FLAG",
        help="extra flag passed verbatim to llama-perplexity (repeatable)",
    )
    parser.add_argument(
        "--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT, metavar="PATH",
        help=f"where to write JSON baselines (default: {DEFAULT_BASELINE_ROOT})",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=None, metavar="PATH",
        help="where to write per-backend logs (default: ~/kernel-work/logs/session-<UTC>/)",
    )
    parser.add_argument(
        "--no-record", action="store_true",
        help="skip writing baseline JSON (dry run)",
    )
    parser.add_argument(
        "--no-sha256", action="store_true",
        help="skip sha256 computation (faster, less reproducible)",
    )
    parser.add_argument(
        "--no-gpu-check", action="store_true",
        help="skip pgrep check for concurrent llama-perplexity",
    )
    parser.add_argument(
        "--wikitext-no-fetch", action="store_true",
        help="fail instead of auto-fetching wikitext if missing",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="echo subprocess output to stderr in real time",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Validate backends
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for b in backends:
        if b not in ("rocm", "vulkan"):
            print(f"error: unknown backend '{b}'; valid: rocm, vulkan", file=sys.stderr)
            return 3

    # Validate model
    args.model = args.model.resolve()
    if not args.model.exists():
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 3
    if not args.model.is_file():
        print(f"error: model path is not a file: {args.model}", file=sys.stderr)
        return 3

    # First-run dir setup
    ensure_dirs(args.baseline_root)

    # Bootstrap wikitext
    bootstrap_wikitext(args.wikitext, allow_fetch=not args.wikitext_no_fetch)

    # Validate wikitext
    if not args.wikitext.exists():
        print(f"error: wikitext not found: {args.wikitext}", file=sys.stderr)
        return 3

    # Resolve log dir
    if args.log_dir is None:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        args.log_dir = Path.home() / "kernel-work" / "logs" / f"session-{ts}"
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # Git info
    git_info = get_git_info()
    host = socket.gethostname()

    # Hashes
    model_sha = None
    wikitext_sha = None
    if not args.no_sha256:
        model_sha = get_model_sha256(args.model)
        wikitext_sha = get_file_sha256(args.wikitext)

    model_stat = args.model.stat()
    timestamp_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Common metadata shared by all backend records
    common_meta = {
        "schema_version": 1,
        "timestamp_utc": timestamp_utc,
        "host": host,
        "git": git_info,
        "model": {
            "path": str(args.model),
            "basename": args.model.name,
            "sha256": model_sha,
            "size_bytes": model_stat.st_size,
            "mtime_unix": int(model_stat.st_mtime),
        },
        "wikitext": {
            "path": str(args.wikitext),
            "sha256": wikitext_sha,
        },
        "harness_version": HARNESS_VERSION,
    }

    # Run backends sequentially
    results = []
    backend_errors = []
    written_paths = []

    for backend in backends:
        print(f"\n[{backend}] starting...", flush=True)
        try:
            r = run_backend(backend, args, args.log_dir)
        except SystemExit as e:
            print(str(e), file=sys.stderr)
            backend_errors.append(backend)
            continue

        results.append(r)

        if r["result"]["ppl"] is None:
            print(
                f"[{backend}] WARNING: could not parse PPL from output. "
                f"Check log: {r['log_path']}",
                file=sys.stderr,
            )
            backend_errors.append(backend)
        elif r["result"]["exit_code"] != 0:
            print(
                f"[{backend}] WARNING: llama-perplexity exited {r['result']['exit_code']}. "
                f"Check log: {r['log_path']}",
                file=sys.stderr,
            )
            backend_errors.append(backend)

        # Write baseline
        if not args.no_record:
            record = {**common_meta, **r}
            bpath = baseline_path(
                args.baseline_root, args.model.name, git_info, host, backend, args.chunks
            )
            write_baseline(bpath, record)
            written_paths.append(bpath)

    # Parity gate
    verdict_str = "(single backend; no parity check)"
    verdict_code = 0

    good_results = [r for r in results if r["result"]["ppl"] is not None and r["result"]["exit_code"] == 0]

    if len(good_results) >= 2:
        ppl_a = good_results[0]["result"]["ppl"]
        ppl_b = good_results[1]["result"]["ppl"]
        label_a = good_results[0]["backend"]
        label_b = good_results[1]["backend"]
        verdict, delta = parity_verdict(ppl_a, ppl_b)
        if verdict == "ERROR":
            verdict_str = "ERROR (invalid PPL values)"
            verdict_code = 2
        else:
            verdict_str = f"{verdict}  (delta = {delta:.2f}%  {label_a} vs {label_b})"
            if verdict == "FAIL":
                verdict_code = 1

    print_summary(
        results=results,
        model_path=args.model,
        wikitext_path=args.wikitext,
        git_info=git_info,
        host=host,
        chunks=args.chunks,
        baseline_paths=written_paths,
        verdict_str=verdict_str,
        model_sha=model_sha,
        wikitext_sha=wikitext_sha,
    )

    if backend_errors:
        print(
            f"\nWARNING: {len(backend_errors)} backend(s) failed or had parse errors: "
            + ", ".join(backend_errors),
            file=sys.stderr,
        )
        return 2

    return verdict_code


if __name__ == "__main__":
    sys.exit(main())
