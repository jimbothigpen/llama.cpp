#!/usr/bin/env python3
"""Answer-accuracy scoring for the EpiCache conv-QA harness.

Two metrics, both standard for long-conversational QA (LoCoMo / LongMemEval):

  * EM  — normalized exact match (SQuAD-style normalization: lowercase, strip
          articles/punctuation/extra whitespace). 1.0 if the normalized
          prediction equals the normalized gold answer, else 0.0.
  * F1  — token-level F1 over the normalized token bags (SQuAD-style). This is
          the headline number EpiCache reports as "LongConvQA accuracy".

A coarser `contains` signal (gold normalized substring of prediction) is also
returned because short-form models often wrap the right fact in a sentence.

Adversarial questions (LoCoMo category 5) have no answerable gold; the desired
behaviour is *abstention*. `score_adversarial()` returns 1.0 when the model
declines to answer (matches an abstain phrase) and 0.0 otherwise.

No third-party deps — stdlib only, so the harness runs on any host with python3.
An optional LLM-judge hook lives in run_eval.py; this module is the offline
default and the unit-test target (`python3 score.py --selftest`).
"""
import re
import string
import sys

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TBL = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """SQuAD normalization: lower, drop punctuation, drop articles, squash ws."""
    if text is None:
        return ""
    text = str(text).lower()
    text = text.translate(_PUNCT_TBL)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def contains(pred: str, gold: str) -> float:
    g = normalize(gold)
    if not g:
        return 0.0
    return 1.0 if g in normalize(pred) else 0.0


def f1(pred: str, gold: str) -> float:
    """Token-level F1 over normalized bags (SQuAD definition)."""
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    # multiset intersection
    common = {}
    gcount = {}
    for t in g_toks:
        gcount[t] = gcount.get(t, 0) + 1
    overlap = 0
    pcount = {}
    for t in p_toks:
        pcount[t] = pcount.get(t, 0) + 1
    for t, c in pcount.items():
        overlap += min(c, gcount.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_toks)
    recall = overlap / len(g_toks)
    return 2 * precision * recall / (precision + recall)


# Phrases that count as a model declining to answer (adversarial / unanswerable).
_ABSTAIN_PATTERNS = [
    "no information", "not mentioned", "not stated", "not specified",
    "does not mention", "doesn't mention", "cannot determine", "can't determine",
    "no answer", "not enough information", "not provided", "unknown",
    "no relevant", "not discussed", "did not", "didn't", "no mention",
    "not in the conversation", "not been mentioned", "cannot be determined",
]


def score_adversarial(pred: str) -> float:
    """1.0 if the model abstained (desired on unanswerable questions), else 0.0."""
    p = normalize(pred)
    return 1.0 if any(normalize(a) in p for a in _ABSTAIN_PATTERNS) else 0.0


def score_one(pred: str, gold: str, category=None) -> dict:
    """Score a single prediction. Category 5 (LoCoMo adversarial) uses abstention."""
    if category == 5:
        a = score_adversarial(pred)
        return {"em": a, "f1": a, "contains": a, "adversarial": True}
    return {
        "em": exact_match(pred, gold),
        "f1": f1(pred, gold),
        "contains": contains(pred, gold),
        "adversarial": False,
    }


def _selftest() -> int:
    cases = [
        # pred, gold, cat, expect_em, expect_f1>0
        ("7 May 2023", "7 May 2023", 4, 1.0, True),
        ("It was on the 7th of May, 2023.", "7 May 2023", 4, 0.0, True),
        ("Paris", "London", 4, 0.0, False),
        ("The answer is the LGBTQ support group", "LGBTQ support group", 1, 0.0, True),
    ]
    ok = True
    for pred, gold, cat, exp_em, exp_f1pos in cases:
        s = score_one(pred, gold, cat)
        if abs(s["em"] - exp_em) > 1e-9:
            print(f"FAIL em: {pred!r} vs {gold!r} -> {s['em']} (exp {exp_em})"); ok = False
        if (s["f1"] > 0) != exp_f1pos:
            print(f"FAIL f1: {pred!r} vs {gold!r} -> {s['f1']} (exp >0: {exp_f1pos})"); ok = False
    # adversarial
    if score_adversarial("There is no information about that in the conversation.") != 1.0:
        print("FAIL adversarial abstain"); ok = False
    if score_adversarial("She realized self-care is important.") != 0.0:
        print("FAIL adversarial non-abstain"); ok = False
    print("score.py selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
