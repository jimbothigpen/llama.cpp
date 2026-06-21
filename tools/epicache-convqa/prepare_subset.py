#!/usr/bin/env python3
"""Build a small, *deterministic* conv-QA subset from a raw benchmark file.

Output is a normalized, harness-internal JSON so run_eval.py is benchmark-agnostic:

    {
      "benchmark": "locomo",
      "conversations": [
        {
          "conv_id": "<sample_id>",
          "context": "<flattened multi-session dialogue, dated>",
          "approx_tokens": <int>,
          "questions": [
            {"qid": "...", "question": "...", "answer": "...", "category": <int>}
          ]
        }
      ]
    }

Selection is fully deterministic (no RNG): the first --convs conversations, and within
each conversation a category-stratified, sorted slice of --qpc questions. Re-running with
the same flags yields byte-identical output -> reproducible gate.

LongMemEval support: pass --format longmemeval with a LongMemEval json; the loader maps
its (haystack_sessions, question, answer) schema onto the same internal shape. Only the
LoCoMo loader is exercised by the committed baselines; the LongMemEval branch is provided
so a later worker can switch benchmarks without touching run_eval.py.
"""
import argparse
import json
import sys
from collections import defaultdict


def _flatten_locomo(conv: dict) -> str:
    """Render LoCoMo sessions into a dated, speaker-tagged transcript."""
    lines = []
    i = 1
    while f"session_{i}" in conv:
        sess = conv[f"session_{i}"]
        date = conv.get(f"session_{i}_date_time", "")
        lines.append(f"=== Session {i}{(' (' + date + ')') if date else ''} ===")
        for turn in sess:
            spk = turn.get("speaker", "?")
            txt = turn.get("text", "")
            # carry image captions as parenthetical context if present (multimodal turns)
            cap = turn.get("blip_caption") or turn.get("caption")
            if cap:
                txt = f"{txt} [shared an image: {cap}]" if txt else f"[shared an image: {cap}]"
            lines.append(f"{spk}: {txt}")
        i += 1
    return "\n".join(lines)


def _stratified(questions, qpc):
    """Deterministic category-stratified slice: round-robin over sorted categories."""
    by_cat = defaultdict(list)
    for q in questions:
        by_cat[q["category"]].append(q)
    for c in by_cat:
        by_cat[c].sort(key=lambda q: q["question"])
    out, cats = [], sorted(by_cat)
    idx = {c: 0 for c in cats}
    while len(out) < qpc and any(idx[c] < len(by_cat[c]) for c in cats):
        for c in cats:
            if idx[c] < len(by_cat[c]):
                out.append(by_cat[c][idx[c]])
                idx[c] += 1
                if len(out) >= qpc:
                    break
    return out


def load_locomo(path, convs, qpc, include_adversarial):
    raw = json.load(open(path))
    out = []
    for s in raw[:convs]:
        qa = s.get("qa", [])
        norm_q = []
        for j, q in enumerate(qa):
            cat = q.get("category")
            if cat == 5 and not include_adversarial:
                continue
            ans = q.get("answer", q.get("adversarial_answer", ""))
            norm_q.append({
                "qid": f"{s.get('sample_id','conv')}_q{j}",
                "question": q["question"],
                "answer": "" if cat == 5 else str(ans),
                "category": cat,
            })
        sel = _stratified(norm_q, qpc)
        ctx = _flatten_locomo(s["conversation"])
        out.append({
            "conv_id": s.get("sample_id", f"conv{len(out)}"),
            "context": ctx,
            "approx_tokens": len(ctx) // 4,
            "questions": sel,
        })
    return out


def load_longmemeval(path, convs, qpc, include_adversarial):
    """Adapter for LongMemEval (xiaowu0162/LongMemEval) json. Best-effort schema map."""
    raw = json.load(open(path))
    items = raw if isinstance(raw, list) else raw.get("data", [])
    out = []
    for s in items[:convs]:
        sessions = s.get("haystack_sessions") or s.get("sessions") or []
        lines = []
        for k, sess in enumerate(sessions, 1):
            lines.append(f"=== Session {k} ===")
            for turn in sess:
                role = turn.get("role", turn.get("speaker", "?"))
                txt = turn.get("content", turn.get("text", ""))
                lines.append(f"{role}: {txt}")
        ctx = "\n".join(lines)
        q = {
            "qid": str(s.get("question_id", f"lme{len(out)}")),
            "question": s.get("question", ""),
            "answer": str(s.get("answer", "")),
            "category": 4,
        }
        out.append({
            "conv_id": str(s.get("question_id", f"lme{len(out)}")),
            "context": ctx,
            "approx_tokens": len(ctx) // 4,
            "questions": [q][:qpc],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw benchmark json (e.g. data/locomo10.json)")
    ap.add_argument("--out", required=True, help="output subset json")
    ap.add_argument("--format", choices=["locomo", "longmemeval"], default="locomo")
    ap.add_argument("--convs", type=int, default=2, help="number of conversations")
    ap.add_argument("--qpc", type=int, default=10, help="questions per conversation")
    ap.add_argument("--include-adversarial", action="store_true",
                    help="include LoCoMo category-5 (abstention) questions")
    args = ap.parse_args()

    loader = {"locomo": load_locomo, "longmemeval": load_longmemeval}[args.format]
    convs = loader(args.raw, args.convs, args.qpc, args.include_adversarial)
    blob = {"benchmark": args.format, "conversations": convs}
    json.dump(blob, open(args.out, "w"), indent=1, ensure_ascii=False)

    nq = sum(len(c["questions"]) for c in convs)
    toks = [c["approx_tokens"] for c in convs]
    print(f"wrote {args.out}: {len(convs)} conversations, {nq} questions")
    print(f"  approx context tokens: min={min(toks)} max={max(toks)} "
          f"mean={sum(toks)//len(toks)}")
    cats = defaultdict(int)
    for c in convs:
        for q in c["questions"]:
            cats[q["category"]] += 1
    print(f"  category distribution: {dict(sorted(cats.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
