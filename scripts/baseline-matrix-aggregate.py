#!/usr/bin/env python3
"""baseline-matrix-aggregate.py — collate per-cell JSONs into matrix.csv + matrix.md.

Recursive scan of <output-dir>, parses every *.json result, builds a tidy
CSV plus a markdown table grouped by (host, backend) for fast delta review.

Output:
  <output-dir>/matrix.csv
  <output-dir>/matrix.md
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    rows: list[dict[str, Any]] = []
    for js in sorted(args.output_dir.glob("*.json")):
        if js.name in ("matrix.csv", "matrix.md"):
            continue
        try:
            d = json.loads(js.read_text())
        except Exception as exc:
            print(f"SKIP {js}: {exc}")
            continue
        ppl_block = d.get("ppl") or {}
        bench_block = d.get("bench") or {}
        rows.append({
            "host":             d.get("host"),
            "backend":          d.get("backend"),
            "weight_quant":     (d.get("model") or {}).get("weight_quant_tag"),
            "ctk":              (d.get("kv_cache") or {}).get("ctk"),
            "ctv":              (d.get("kv_cache") or {}).get("ctv"),
            "adaptive_mode":    d.get("adaptive_mode"),
            "ppl":              ppl_block.get("ppl"),
            "ppl_stddev":       ppl_block.get("stddev"),
            "ppl_chunks":       d.get("ppl_chunks"),
            "ppl_ctx":          d.get("ppl_ctx"),
            "pp_tps":           bench_block.get("pp_tps"),
            "pp_stddev":        bench_block.get("pp_stddev"),
            "pp_n":             bench_block.get("pp_n"),
            "tg_tps":           bench_block.get("tg_tps"),
            "tg_stddev":        bench_block.get("tg_stddev"),
            "tg_n":             bench_block.get("tg_n"),
            "size_bytes":       (d.get("model") or {}).get("size_bytes"),
            "imatrix":          bool(d.get("imatrix")),
            "commit":           (d.get("git") or {}).get("commit"),
            "branch":           (d.get("git") or {}).get("branch"),
            "timestamp_utc":    d.get("timestamp_utc"),
            "source_json":      js.name,
        })

    if not rows:
        print("No result JSONs found.")
        return 0

    csv_path = args.output_dir / "matrix.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")

    # Markdown: per-cell table, with delta-from-best columns.
    md: list[str] = ["# Baseline matrix\n"]
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        by_cell.setdefault((r["host"], r["backend"]), []).append(r)

    for (host, backend), rs in sorted(by_cell.items()):
        md.append(f"\n## {host} / {backend}\n")
        ppls = [r["ppl"] for r in rs if r["ppl"] is not None]
        pps  = [r["pp_tps"] for r in rs if r["pp_tps"] is not None]
        tgs  = [r["tg_tps"] for r in rs if r["tg_tps"] is not None]
        best_ppl = min(ppls) if ppls else None  # lower is better
        best_pp  = max(pps) if pps else None    # higher is better
        best_tg  = max(tgs) if tgs else None

        md.append("| Weight | KV (K/V) | PPL | dPPL%vs_best | "
                  "PP tps | dPP%vs_best | TG tps | dTG%vs_best | imatrix |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        # Sort within cell: weight quant, then ctk, then ctv
        for r in sorted(rs, key=lambda x: (x["weight_quant"] or "", x["ctk"] or "", x["ctv"] or "", x.get("adaptive_mode") or 0)):
            dppl = (f"{(r['ppl']-best_ppl)/best_ppl*100:+.2f}%"
                    if r["ppl"] is not None and best_ppl else "")
            dpp  = (f"{(r['pp_tps']-best_pp)/best_pp*100:+.2f}%"
                    if r["pp_tps"] is not None and best_pp else "")
            dtg  = (f"{(r['tg_tps']-best_tg)/best_tg*100:+.2f}%"
                    if r["tg_tps"] is not None and best_tg else "")
            ppl_str = f"{r['ppl']:.4f}" if r['ppl'] is not None else "—"
            pp_str  = f"{r['pp_tps']:.2f}" if r['pp_tps'] is not None else "—"
            tg_str  = f"{r['tg_tps']:.2f}" if r['tg_tps'] is not None else "—"
            la = r.get("adaptive_mode")
            kv_label = f"{r['ctk']}/{r['ctv']}" + (f"  la{la}" if la else "")
            md.append(f"| {r['weight_quant']} | {kv_label} | "
                      f"{ppl_str} | {dppl} | {pp_str} | {dpp} | "
                      f"{tg_str} | {dtg} | "
                      f"{'yes' if r['imatrix'] else 'no'} |")

    md_path = args.output_dir / "matrix.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
