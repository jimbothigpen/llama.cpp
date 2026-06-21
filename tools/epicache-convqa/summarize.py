#!/usr/bin/env python3
"""Combine per-arm result JSONs into one comparison table + the P1-vs-full gap.

Usage:
    summarize.py results_full.json results_p1.json [results_plain-evict.json ...]

Prints a markdown table (overall EM/F1/contains per arm) and the headline number
this whole harness exists to produce: the accuracy gap between full-attention and
each compressed arm — i.e. how much accuracy episodic routing (P3) must recover.
"""
import json
import sys


def overall(items):
    n = len(items) or 1
    return {
        "n": len(items),
        "em": sum(r["em"] for r in items) / n,
        "f1": sum(r["f1"] for r in items) / n,
        "contains": sum(r["contains"] for r in items) / n,
    }


def main(paths):
    arms = {}
    for p in paths:
        d = json.load(open(p))
        arms[d["arm"]] = {"agg": overall(d["items"]),
                          "model": d.get("model"), "budget": d.get("budget")}
    if not arms:
        print("no inputs"); return 1

    model = next(iter(arms.values()))["model"]
    print(f"\n## EpiCache conv-QA baseline comparison (model={model})\n")
    print("| arm | n | EM | F1 | contains |")
    print("|---|---:|---:|---:|---:|")
    order = [a for a in ["full", "p1", "plain-evict", "eq3"] if a in arms]
    for a in order:
        g = arms[a]["agg"]
        bud = f" (M={arms[a]['budget']})" if a in ("p1",) else ""
        print(f"| {a}{bud} | {g['n']} | {g['em']:.3f} | {g['f1']:.3f} | {g['contains']:.3f} |")

    if "full" in arms:
        f = arms["full"]["agg"]
        print("\n### Accuracy gap vs full-attention (F1)\n")
        print("| arm | F1 | gap vs full | rel. retention |")
        print("|---|---:|---:|---:|")
        for a in order:
            if a == "full":
                continue
            g = arms[a]["agg"]
            gap = f["f1"] - g["f1"]
            ret = (g["f1"] / f["f1"]) if f["f1"] > 0 else float("nan")
            print(f"| {a} | {g['f1']:.3f} | {gap:+.3f} | {ret:.1%} |")
        print("\n> The P1-vs-full gap is the accuracy episodic multi-cache routing (P3) "
              "must recover. A near-zero gap means P1's prefill bounding already "
              "preserves conv-QA accuracy at this budget; a large gap quantifies the "
              "headroom P3 targets.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
