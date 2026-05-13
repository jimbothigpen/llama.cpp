#!/usr/bin/env python3
"""baseline-matrix.py — single-cell PPL + bench runner for yggdrasil baseline matrix.

Runs llama-perplexity + llama-bench for one (model, host, backend, kv-config)
cell and writes a structured JSON result to the matrix output directory.

See: session-03-baseline-matrix-design.md for the full plan.

Usage:
    baseline-matrix.py --model <path> --host ai00|ai01 --backend rocm|vulkan
                       --task ppl|bench|both [--ctk f16] [--ctv f16]
                       [--imatrix <path>] --output-dir <dir>
                       [--ppl-chunks 25] [--ppl-ctx 512]
                       [--weight-quant-tag Q4_K_M] [--force] [--dry-run]
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


REPO_ROOT = Path("/mnt/cephfs/0/Container/systems/ai00/users/builduser/llama-yggdrasil")
WIKITEXT_PATH = Path(
    "/mnt/cephfs/0/Container/systems/ai00/users/builduser/yggdrasil-context/"
    "corpora/wikitext-2-raw-test.txt"
)
SSH_PORT_AI01 = "2229"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_meta(repo: Path) -> dict:
    def run(*args):
        return subprocess.check_output(
            ["git", "-C", str(repo)] + list(args), text=True
        ).strip()
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def binary_path(backend: str, tool: str) -> str:
    return f"/opt/llama-yggdrasil-{backend}/bin/{tool}"


def env_for(host: str, backend: str) -> dict[str, str]:
    env: dict[str, str] = {}
    if host == "ai01" and backend == "rocm":
        env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.2"
    if host == "ai01" and backend == "vulkan":
        env["GGML_VK_PREFER_HOST_MEMORY"] = "1"
    return env


def run_remote(host: str, backend: str, cmd: list[str], timeout: int) -> tuple[str, int]:
    """Run `cmd` on `host`. Local for ai00, ssh for ai01. Returns (combined_output, rc)."""
    env = env_for(host, backend)
    env_prefix = [f"{k}={shlex.quote(v)}" for k, v in env.items()]
    if host == "ai00":
        os_env = os.environ.copy()
        os_env.update(env)
        r = subprocess.run(
            cmd, capture_output=True, text=True, env=os_env, timeout=timeout
        )
        return (r.stdout or "") + (r.stderr or ""), r.returncode
    else:
        remote_cmd = " ".join(env_prefix + [shlex.quote(c) for c in cmd])
        full = ["ssh", "-p", SSH_PORT_AI01, host, remote_cmd]
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or ""), r.returncode


PPL_FINAL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*(\d+\.\d+)\s*\+/-\s*(\d+\.\d+)")
# llama-bench markdown row format varies; capture by looking for "pp512" / "tg128" rows
BENCH_PP_RE = re.compile(r"\|\s*pp(\d+)\s*\|\s*([\d.]+)\s*[±]\s*([\d.]+)\s*\|")
BENCH_TG_RE = re.compile(r"\|\s*tg(\d+)\s*\|\s*([\d.]+)\s*[±]\s*([\d.]+)\s*\|")


def run_perplexity(
    host: str, backend: str, model: Path, ctk: str, ctv: str,
    imatrix: Optional[Path], chunks: int, ctx: int, timeout: int,
) -> tuple[Optional[dict], str, int]:
    cmd = [
        binary_path(backend, "llama-perplexity"),
        "-m", str(model),
        "-f", str(WIKITEXT_PATH),
        "-c", str(ctx),
        "--chunks", str(chunks),
        "-ngl", "999",
        "-fa", "on",
        "--no-mmap",
        "-ctk", ctk,
        "-ctv", ctv,
    ]
    out, rc = run_remote(host, backend, cmd, timeout)
    m = PPL_FINAL_RE.search(out)
    if m is None:
        return None, out, rc
    return {
        "ppl": float(m.group(1)),
        "stddev": float(m.group(2)),
        "chunks": chunks,
        "ctx": ctx,
    }, out, rc


def run_bench(
    host: str, backend: str, model: Path, ctk: str, ctv: str, timeout: int,
) -> tuple[Optional[dict], str, int]:
    # Wrap under the per-host advisory file lock so concurrent compile
    # jobs don't skew throughput numbers. run_perplexity() is
    # compile-tolerant and does NOT use the wrapper.
    cmd = [
        str(REPO_ROOT / "scripts/with-bench-mutex.sh"),
        binary_path(backend, "llama-bench"),
        "-m", str(model),
        "-ngl", "999",
        "-fa", "1",
        "-mmp", "0",
        "-ctk", ctk,
        "-ctv", ctv,
    ]
    out, rc = run_remote(host, backend, cmd, timeout)
    pp = BENCH_PP_RE.search(out)
    tg = BENCH_TG_RE.search(out)
    res: dict = {}
    if pp:
        res["pp_n"] = int(pp.group(1))
        res["pp_tps"] = float(pp.group(2))
        res["pp_stddev"] = float(pp.group(3))
    if tg:
        res["tg_n"] = int(tg.group(1))
        res["tg_tps"] = float(tg.group(2))
        res["tg_stddev"] = float(tg.group(3))
    return (res or None), out, rc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--host", required=True, choices=["ai00", "ai01"])
    p.add_argument("--backend", required=True, choices=["rocm", "vulkan"])
    p.add_argument("--task", required=True, choices=["ppl", "bench", "both"])
    p.add_argument("--ctk", default="f16")
    p.add_argument("--ctv", default="f16")
    p.add_argument("--imatrix", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--ppl-chunks", type=int, default=25)
    p.add_argument("--ppl-ctx", type=int, default=512)
    p.add_argument("--weight-quant-tag")
    p.add_argument("--ppl-timeout", type=int, default=3600,
                   help="seconds before killing llama-perplexity")
    p.add_argument("--bench-timeout", type=int, default=1800,
                   help="seconds before killing llama-bench")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if matching result file exists")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    weight_tag = args.weight_quant_tag or args.model.stem.split("-")[-1]
    if args.ctk == args.ctv:
        kv_tag = f"kv-{args.ctk}"
    else:
        kv_tag = f"kv-{args.ctk}-{args.ctv}"
    cell_id = f"{args.host}__{args.backend}"
    config_id = f"weight-{weight_tag}__{kv_tag}"
    out_path = args.output_dir / f"{cell_id}__{config_id}.json"

    if out_path.exists() and not args.force:
        print(f"SKIP existing: {out_path}", file=sys.stderr)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    if args.dry_run:
        print(f"DRY-RUN host={args.host} backend={args.backend} "
              f"weight={weight_tag} ctk={args.ctk} ctv={args.ctv} "
              f"-> {out_path}")
        return 0

    record: dict = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "backend": args.backend,
        "git": git_meta(REPO_ROOT),
        "binary_dir": f"/opt/llama-yggdrasil-{args.backend}/bin",
        "model": {
            "path": str(args.model),
            "basename": args.model.name,
            "size_bytes": args.model.stat().st_size,
            "weight_quant_tag": weight_tag,
        },
        "kv_cache": {"ctk": args.ctk, "ctv": args.ctv},
        "imatrix": ({"path": str(args.imatrix)} if args.imatrix else None),
        "ppl_chunks": args.ppl_chunks,
        "ppl_ctx": args.ppl_ctx,
        "wikitext_path": str(WIKITEXT_PATH),
        "env": env_for(args.host, args.backend),
    }

    if args.task in ("ppl", "both"):
        print(f"PPL  {cell_id} {config_id}...", file=sys.stderr)
        ppl, log, rc = run_perplexity(
            args.host, args.backend, args.model, args.ctk, args.ctv,
            args.imatrix, args.ppl_chunks, args.ppl_ctx, args.ppl_timeout,
        )
        ppl_log = log_dir / f"{cell_id}__{config_id}__ppl.log"
        ppl_log.write_text(log)
        record["ppl"] = ppl
        record["ppl_rc"] = rc
        record["ppl_log_path"] = str(ppl_log)
        print(f"     -> PPL={ppl}" if ppl else f"     -> FAILED rc={rc}",
              file=sys.stderr)

    if args.task in ("bench", "both"):
        print(f"BENCH {cell_id} {config_id}...", file=sys.stderr)
        bench, log, rc = run_bench(
            args.host, args.backend, args.model, args.ctk, args.ctv,
            args.bench_timeout,
        )
        bench_log = log_dir / f"{cell_id}__{config_id}__bench.log"
        bench_log.write_text(log)
        record["bench"] = bench
        record["bench_rc"] = rc
        record["bench_log_path"] = str(bench_log)
        print(f"     -> {bench}" if bench else f"     -> FAILED rc={rc}",
              file=sys.stderr)

    out_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"WROTE {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
