#!/usr/bin/env python3
"""Model-evaluation + tier-classification harness.

Scores ANY already-running OpenAI-compatible chat server on a suite of
capability-dimension probes, aggregates per-dimension scores, and assigns a
deployment **tier** from the policy in ``tiers.py``.  Reusable across the
local-serving fleet (122B / 35B / 27B-Coder / 9B-Coder / 4B-Coder): point it at
a base URL + model id and it emits a scorecard.

Unlike the EpiCache conv-QA harness (which *launches* a server per KV-policy
arm), this harness only *queries* an existing endpoint — so it never disrupts a
production server.  Queries are sequential by default (``--rps`` throttles) to
keep load modest.

Usage:
    python3 eval_harness.py \\
        --base-url http://127.0.0.1:8083 --model qwen36-35b \\
        --suite suites/default.json \\
        --out results/qwen36-35b.json --md results/qwen36-35b.md

Output: a results JSON (config + per-item predictions/scores + per-dimension
aggregates + tier) and an optional markdown scorecard.  The JSON is the source
of truth; the markdown is a human view rendered from it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import checks  # noqa: E402
import tiers   # noqa: E402


# --------------------------------------------------------------------- server io
def chat(base, model, system, user, max_tokens, timeout, no_think=True):
    """One OpenAI-compat /v1/chat/completions call. Returns (text, dt, usage).

    Greedy (temperature 0, top_k 1) and cache_prompt:false for reproducibility.
    Disables reasoning-model chain-of-thought so the answer lands in `content`;
    falls back to reasoning_content if a thinking model emptied content anyway.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_k": 1,
        "max_tokens": max_tokens,
        "cache_prompt": False,
        "stream": False,
    }
    if no_think:
        # Harmless on non-reasoning models (unknown kwargs ignored by template).
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    dt = time.time() - t0
    msg = resp["choices"][0]["message"]
    text = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()
    return text, dt, resp.get("usage", {})


def server_info(base, model, timeout=10):
    """Best-effort /v1/models lookup for stamping the scorecard."""
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/v1/models", timeout=timeout) as r:
            data = json.loads(r.read())
        for m in data.get("data", []):
            if m.get("id") == model:
                return m.get("meta", {})
    except Exception:
        pass
    return {}


# ----------------------------------------------------------------------- runner
def run_suite(args):
    suite = json.load(open(args.suite))
    defaults = suite.get("defaults", {})
    items = suite["items"]
    if args.dimensions:
        want = set(args.dimensions.split(","))
        items = [it for it in items if it["dimension"] in want]
    if args.max_items:
        items = items[: args.max_items]

    results = []
    min_interval = 1.0 / args.rps if args.rps > 0 else 0.0
    last = 0.0
    for i, it in enumerate(items):
        now = time.time()
        wait = min_interval - (now - last)
        if wait > 0:
            time.sleep(wait)
        last = time.time()

        system = it.get("system", defaults.get("system", ""))
        max_tokens = it.get("max_tokens", defaults.get("max_tokens", 512))
        pred, dt, usage, err = "", 0.0, {}, None
        try:
            pred, dt, usage = chat(args.base_url, args.model, system, it["prompt"],
                                   max_tokens, args.req_timeout, no_think=not args.thinking)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        if err:
            res = checks.Result(0.0, False, f"request-failed: {err}")
        else:
            res = checks.score_item(pred, it, allow_code_exec=args.allow_code_exec)

        results.append({
            "id": it["id"], "dimension": it["dimension"], "check": it["check"],
            "prompt": it["prompt"], "pred": pred, "score": res.score,
            "ok": res.ok, "detail": res.detail, "latency_s": round(dt, 2),
            "usage": usage, "error": err,
        })
        flag = "ok " if res.ok else ("ERR" if err else "x  ")
        print(f"[{i+1:>2}/{len(items)}] {flag} {it['dimension']:<22} {it['id']:<10} "
              f"score={res.score:.2f} {res.detail[:60]}", flush=True)

    return suite, results


def aggregate(results):
    """Per-dimension mean score + overall counts."""
    bydim = {}
    for r in results:
        bydim.setdefault(r["dimension"], []).append(r["score"])
    dim_scores = {d: sum(v) / len(v) for d, v in bydim.items()}
    dim_detail = {d: {"n": len(v), "mean": sum(v) / len(v)} for d, v in bydim.items()}
    return dim_scores, dim_detail


# -------------------------------------------------------------------- scorecard
def render_md(scorecard):
    sc = scorecard
    L = []
    L.append(f"# Model eval scorecard — {sc['model']}")
    L.append("")
    L.append(f"- endpoint: `{sc['base_url']}`")
    L.append(f"- suite: `{sc['suite']}` ({sc['n_items']} items)")
    if sc.get("server_meta"):
        meta = sc["server_meta"]
        L.append(f"- server: n_params={meta.get('n_params')} n_ctx_train={meta.get('n_ctx_train')} "
                 f"size={meta.get('size')}")
    L.append(f"- policy: `{sc['policy_version']}`")
    L.append("")
    L.append(f"## TIER: **{sc['tier']}** ({sc['tier_label']}) — composite **{sc['composite']:.3f}**")
    L.append("")
    L.append(f"> routing: {sc['tier_routing']}")
    L.append("")
    L.append("### Per-dimension")
    L.append("")
    L.append("| dimension | n | mean score |")
    L.append("|---|---:|---:|")
    for d in sorted(sc["dim_detail"]):
        dd = sc["dim_detail"][d]
        L.append(f"| {d} | {dd['n']} | {dd['mean']:.3f} |")
    L.append(f"| **composite** |  | **{sc['composite']:.3f}** |")
    L.append("")
    if sc.get("tier_reasons"):
        L.append("### Why not a higher tier")
        L.append("")
        for r in sc["tier_reasons"]:
            L.append(f"- {r}")
        L.append("")
    L.append("### Tier policy")
    L.append("")
    L.append(tiers.threshold_table_md())
    L.append("")
    L.append("### Failed / notable items")
    L.append("")
    L.append("| id | dim | score | detail |")
    L.append("|---|---|---:|---|")
    for r in sc["items"]:
        if not r["ok"]:
            L.append(f"| {r['id']} | {r['dimension']} | {r['score']:.2f} | {r['detail'][:70]} |")
    L.append("")
    return "\n".join(L)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="OpenAI-compat base, e.g. http://127.0.0.1:8083")
    ap.add_argument("--model", required=True, help="served model id (from /v1/models)")
    ap.add_argument("--suite", default=os.path.join(HERE, "suites/default.json"))
    ap.add_argument("--out", default="", help="results JSON path")
    ap.add_argument("--md", default="", help="markdown scorecard path")
    ap.add_argument("--dimensions", default="", help="comma list to restrict dims")
    ap.add_argument("--max-items", type=int, default=0, help="cap items (0=all)")
    ap.add_argument("--rps", type=float, default=1.0, help="max requests/sec (0=unlimited)")
    ap.add_argument("--req-timeout", type=float, default=180.0, help="per-request timeout s")
    ap.add_argument("--thinking", action="store_true", help="allow reasoning CoT (default off)")
    ap.add_argument("--allow-code-exec", dest="allow_code_exec", action="store_true", default=True)
    ap.add_argument("--no-code-exec", dest="allow_code_exec", action="store_false")
    args = ap.parse_args(argv)

    suite, results = run_suite(args)
    dim_scores, dim_detail = aggregate(results)
    tier, comp, reasons = tiers.classify(dim_scores)
    meta = server_info(args.base_url, args.model)

    scorecard = {
        "harness": "model-eval-framework",
        "model": args.model,
        "base_url": args.base_url,
        "suite": suite.get("suite", os.path.basename(args.suite)),
        "n_items": len(results),
        "server_meta": meta,
        "policy_version": tiers.POLICY["version"],
        "dim_scores": dim_scores,
        "dim_detail": dim_detail,
        "composite": comp,
        "tier": tier["tier"],
        "tier_label": tier["label"],
        "tier_routing": tier["routing"],
        "tier_reasons": reasons,
        "items": results,
    }

    print()
    print(f"=== {args.model}: TIER {tier['tier']} ({tier['label']}) "
          f"composite={comp:.3f} ===")
    for d in sorted(dim_detail):
        print(f"  {d:<24} {dim_detail[d]['mean']:.3f}  (n={dim_detail[d]['n']})")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(scorecard, open(args.out, "w"), indent=1, ensure_ascii=False)
        print(f"wrote {args.out}")
    if args.md:
        os.makedirs(os.path.dirname(os.path.abspath(args.md)), exist_ok=True)
        open(args.md, "w").write(render_md(scorecard))
        print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
