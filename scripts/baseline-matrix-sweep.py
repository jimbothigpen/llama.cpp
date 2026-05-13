#!/usr/bin/env python3
"""baseline-matrix-sweep.py — driver for full weight-quant or kv-quant baseline sweep.

Enumerates cells × rows defined in session-03-baseline-matrix-design.md and
shells out to baseline-matrix.py for each.

Concurrency policy: one llama-perplexity/bench job per host at a time;
ai00 and ai01 may run concurrently. This driver dispatches sequentially
within a single host but supports per-host parallel invocations:

    # In two separate shells:
    baseline-matrix-sweep.py --sweep weight --cells ai00:rocm ...
    baseline-matrix-sweep.py --sweep weight --cells ai01:rocm ...
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

# Weight quants for the baseline (per design doc §4.1).
# Excludes F32, MXFP4_MOE, TQ1_0, TQ2_0.
WEIGHT_QUANTS = [
    "F16", "BF16", "Q8_0", "Q6_K",
    "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0",
    "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0",
    "IQ4_XS", "IQ4_NL",
    "Q3_K_L", "Q3_K_M", "Q3_K_S",
    "IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS",
    "Q2_K", "Q2_K_S",
    "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS",
    "IQ1_M", "IQ1_S",
    "Q1_0",
]

# Quants that need (or strongly want) the imatrix.
# k-quants (Q*_K) accept imatrix optionally; convention is always-on (per design §4.1).
NEEDS_IMATRIX = lambda q: q.startswith("IQ") or "_K" in q

# Effective bpw per element for mainline KV cache types (excluding f32).
KV_BPW: dict[str, float] = {
    "f16":    16.00,
    "bf16":   16.00,
    "q8_0":    8.50,
    "q5_1":    6.06,
    "q5_0":    5.56,
    "q4_1":    5.06,
    "iq4_nl":  4.50,
    "q4_0":    4.50,
}

CELLS = [
    ("ai00", "rocm"),
    ("ai00", "vulkan"),
    ("ai01", "rocm"),
    ("ai01", "vulkan"),
]


def sensible_kv_combos(skip_bf16_k: bool = False) -> list[tuple[str, str]]:
    """Pairs (ctk, ctv) where bpw(K) >= bpw(V).  Same-bpw pairs included."""
    pairs: list[tuple[str, str]] = []
    types = list(KV_BPW.keys())
    for k in types:
        if skip_bf16_k and k == "bf16":
            continue
        for v in types:
            if KV_BPW[k] >= KV_BPW[v]:
                pairs.append((k, v))
    return pairs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Directory with the per-quant GGUF files.")
    p.add_argument("--model-base", required=True,
                   help="Basename stem, e.g. Qwen3.5-9B "
                        "(expects <base>-<QUANT>.gguf inside model-dir).")
    p.add_argument("--imatrix", type=Path,
                   help="Path to imatrix.gguf used for IQ/k-quant runs.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Per-cell JSONs go here.")
    p.add_argument("--sweep", choices=["weight", "kv"], required=True)
    p.add_argument("--cells", nargs="+", default=None,
                   help='Subset of cells, e.g. "ai00:rocm ai01:vulkan"')
    p.add_argument("--quants", nargs="+", default=None,
                   help="Subset of weight quants (default: all per WEIGHT_QUANTS).")
    p.add_argument("--fixed-weight", default="Q4_K_M",
                   help="Weight quant for the KV sweep (default Q4_K_M).")
    p.add_argument("--skip-bf16-k", action="store_true",
                   help="For KV sweep: skip K=bf16 rows (redundant with K=f16).")
    p.add_argument("--task", default="both", choices=["ppl", "bench", "both"])
    p.add_argument("--runner", default=str(Path(__file__).parent / "baseline-matrix.py"),
                   help="Path to baseline-matrix.py")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cells = CELLS
    if args.cells:
        cells = []
        for c in args.cells:
            host, backend = c.split(":")
            cells.append((host, backend))

    configs: list[tuple[Path, str, str, str, str, str]] = []
    if args.sweep == "weight":
        quants = args.quants or WEIGHT_QUANTS
        for q in quants:
            model = args.model_dir / f"{args.model_base}-{q}.gguf"
            for host, backend in cells:
                configs.append((model, host, backend, "f16", "f16", q))
    else:
        model = args.model_dir / f"{args.model_base}-{args.fixed_weight}.gguf"
        pairs = sensible_kv_combos(skip_bf16_k=args.skip_bf16_k)
        for host, backend in cells:
            for ctk, ctv in pairs:
                configs.append((model, host, backend, ctk, ctv, args.fixed_weight))

    print(f"Sweep: {args.sweep}  cells={cells}  configs={len(configs)}")

    failures = 0
    for i, (model, host, backend, ctk, ctv, qtag) in enumerate(configs, 1):
        if not model.exists():
            print(f"[{i}/{len(configs)}] MISSING model: {model}", file=sys.stderr)
            failures += 1
            continue
        cmd = [
            sys.executable, args.runner,
            "--model", str(model),
            "--host", host,
            "--backend", backend,
            "--task", args.task,
            "--ctk", ctk,
            "--ctv", ctv,
            "--output-dir", str(args.output_dir),
            "--weight-quant-tag", qtag,
        ]
        if NEEDS_IMATRIX(qtag) and args.imatrix:
            cmd += ["--imatrix", str(args.imatrix)]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.force:
            cmd.append("--force")
        print(f"[{i}/{len(configs)}] {host}/{backend}  weight={qtag}  "
              f"ctk={ctk} ctv={ctv}")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"  -> rc={r.returncode}", file=sys.stderr)
            failures += 1

    print(f"\nDone. {len(configs) - failures}/{len(configs)} OK, "
          f"{failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
