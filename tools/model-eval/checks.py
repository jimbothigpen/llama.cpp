#!/usr/bin/env python3
"""Deterministic, stdlib-only answer checks for the model-eval harness.

Every check takes a model `pred` (string) plus the item's spec and returns a
``Result(score, ok, detail)`` where ``score`` is in [0, 1].  Scores are
deterministic and offline so a run is reproducible from the results JSON alone
(no LLM-judge, no network).  An optional LLM-judge can be layered on top later;
these are the reproducible default.

Normalization (`normalize`) is SQuAD-style and shared by the string checks; it
is lifted from the EpiCache conv-QA scorer so EM/F1 semantics match across
harnesses.

Check types (declared per item as ``"check": "<type>"``):

  exact          normalized exact-match vs ``answer``
  contains       normalized ``answer`` is a substring of the prediction
  any_of         any string in ``answer`` (list) is contained  -> 1.0
  all_of         every string in ``answer`` (list) is contained -> 1.0 (else
                 partial credit = fraction present)
  regex          prediction matches ``pattern`` (re.search, IGNORECASE|DOTALL)
  numeric        last number in prediction == ``answer`` within ``tol``
  abstain        prediction declines / says it cannot answer -> 1.0
                 (for adversarial / unanswerable robustness items)
  code_exec      extract a Python code block, append ``test``, run in a
                 subprocess with a timeout; exit 0 -> 1.0

``code_exec`` runs model-generated code; it is gated by the harness behind
``--allow-code-exec`` (default on for the local pilot, since the box is trusted
and the tasks are self-authored).  All execution is in a separate process with
a wall-clock timeout and no inherited stdin.
"""
from __future__ import annotations

import re
import string
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# ---------------------------------------------------------------- normalization
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TBL = str.maketrans("", "", string.punctuation)


def normalize(text) -> str:
    """SQuAD normalization: lower, drop punctuation, drop articles, squash ws."""
    if text is None:
        return ""
    text = str(text).lower()
    text = text.translate(_PUNCT_TBL)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


# Numbers: optional sign, thousands separators, decimals. Returns the *last*
# numeric token (final-answer convention for chain-of-thought-style replies).
_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")

# Phrases that count as the model declining to answer (adversarial / unanswerable).
_ABSTAIN_PATTERNS = [
    "no information", "not mentioned", "not stated", "not specified",
    "does not mention", "doesn't mention", "cannot determine",
    "can't determine", "cannot answer", "can't answer", "unable to",
    "not enough information", "not provided", "no mention", "i don't know",
    "i do not know", "there is no", "isn't any", "is no information",
    "not possible to", "cannot be determined", "cannot know", "can't know",
    "do not have access", "don't have access", "no access to", "no way to know",
    "have no way", "cannot tell", "can't tell", "not aware of",
]

# ---------------------------------------------------------------------- results
@dataclass
class Result:
    score: float      # in [0, 1]
    ok: bool          # score >= item pass threshold (default 1.0 == fully correct)
    detail: str = ""  # short human-readable note for the scorecard / debugging


def _last_number(text: str):
    matches = _NUM_RE.findall(text or "")
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _is_abstention(pred: str) -> bool:
    p = normalize(pred)
    return any(normalize(pat) in p for pat in _ABSTAIN_PATTERNS)


# ---------------------------------------------------------------- code extract
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(pred: str) -> str:
    """Pull the first fenced code block; fall back to the whole reply."""
    m = _FENCE_RE.search(pred or "")
    if m:
        return m.group(1)
    return pred or ""


def run_code_exec(pred: str, test: str, timeout: float = 10.0):
    """Append ``test`` to the model's code, run in a subprocess. Returns Result."""
    code = extract_code(pred)
    program = code + "\n\n" + test + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as fh:
        fh.write(program)
        fh.flush()
        try:
            proc = subprocess.run(
                [sys.executable, fh.name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return Result(0.0, False, f"timeout>{timeout}s")
        except Exception as e:  # pragma: no cover - defensive
            return Result(0.0, False, f"exec-error:{type(e).__name__}")
    if proc.returncode == 0:
        return Result(1.0, True, "exec ok")
    tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    return Result(0.0, False, f"exit={proc.returncode}: {tail[0][:120]}")


# ------------------------------------------------------------------- dispatch
def score_item(pred: str, item: dict, allow_code_exec: bool = True) -> Result:
    """Score one prediction against one suite item. Pure (except code_exec)."""
    check = item.get("check", "contains")
    ans = item.get("answer")
    p_norm = normalize(pred)

    if check == "exact":
        s = 1.0 if p_norm == normalize(ans) else 0.0
        return Result(s, s >= 1.0, "")

    if check == "contains":
        g = normalize(ans)
        s = 1.0 if g and g in p_norm else 0.0
        return Result(s, s >= 1.0, "")

    if check == "any_of":
        golds = ans if isinstance(ans, list) else [ans]
        hit = next((g for g in golds if normalize(g) and normalize(g) in p_norm), None)
        return Result(1.0 if hit else 0.0, hit is not None, f"hit={hit!r}" if hit else "")

    if check == "all_of":
        golds = ans if isinstance(ans, list) else [ans]
        present = [g for g in golds if normalize(g) and normalize(g) in p_norm]
        frac = len(present) / len(golds) if golds else 0.0
        return Result(frac, frac >= 1.0, f"{len(present)}/{len(golds)} present")

    if check == "regex":
        # DOTALL only — case sensitivity is intentional (some items check
        # ALL-CAPS / exact casing). Items wanting case-insensitivity add the
        # inline (?i) flag to their own pattern.
        pat = item.get("pattern", "")
        m = re.search(pat, pred or "", re.DOTALL)
        return Result(1.0 if m else 0.0, m is not None, "")

    if check == "numeric":
        got = _last_number(pred)
        want = _last_number(str(ans))
        tol = float(item.get("tol", 0.0))
        if got is None or want is None:
            return Result(0.0, False, f"no-number got={got} want={want}")
        ok = abs(got - want) <= tol
        return Result(1.0 if ok else 0.0, ok, f"got={got} want={want}")

    if check == "abstain":
        ab = _is_abstention(pred)
        return Result(1.0 if ab else 0.0, ab, "abstained" if ab else "answered")

    if check == "code_exec":
        if not allow_code_exec:
            return Result(0.0, False, "code-exec disabled")
        return run_code_exec(pred, item.get("test", ""), float(item.get("timeout", 10.0)))

    return Result(0.0, False, f"unknown-check:{check}")


# ---------------------------------------------------------------------- selftest
def _selftest() -> int:
    ok = True

    def expect(cond, msg):
        nonlocal ok
        if not cond:
            print("FAIL:", msg)
            ok = False

    expect(score_item("Paris", {"check": "exact", "answer": "paris"}).score == 1.0, "exact")
    expect(score_item("It is Paris.", {"check": "contains", "answer": "Paris"}).score == 1.0, "contains")
    expect(score_item("42 dogs", {"check": "any_of", "answer": ["cats", "dogs"]}).score == 1.0, "any_of")
    r = score_item("has foo and bar", {"check": "all_of", "answer": ["foo", "bar", "baz"]})
    expect(abs(r.score - 2 / 3) < 1e-9, "all_of partial")
    expect(score_item("```json\n{}\n```", {"check": "regex", "pattern": r"```json"}).score == 1.0, "regex")
    # Case-sensitive ALL-CAPS pattern must NOT be forced case-insensitive.
    expect(score_item("ACKNOWLEDGED", {"check": "regex", "pattern": r"^[^a-z]*$"}).score == 1.0, "regex case-sensitive caps")
    expect(score_item("acknowledged", {"check": "regex", "pattern": r"^[^a-z]*$"}).score == 0.0, "regex case-sensitive lower")
    expect(score_item("I cannot know what you ate.", {"check": "abstain"}).score == 1.0, "abstain cannot-know")
    expect(score_item("I do not have access to that.", {"check": "abstain"}).score == 1.0, "abstain no-access")
    expect(score_item("the total is 17.", {"check": "numeric", "answer": 17}).score == 1.0, "numeric")
    expect(score_item("about 3.14159", {"check": "numeric", "answer": 3.14, "tol": 0.01}).score == 1.0, "numeric tol")
    expect(score_item("There is no information about that.", {"check": "abstain"}).score == 1.0, "abstain yes")
    expect(score_item("The answer is blue.", {"check": "abstain"}).score == 0.0, "abstain no")
    r = score_item("```python\ndef f():\n    return 1\n```",
                   {"check": "code_exec", "test": "assert f() == 1"})
    expect(r.score == 1.0, f"code_exec pass ({r.detail})")
    r = score_item("```python\ndef f():\n    return 2\n```",
                   {"check": "code_exec", "test": "assert f() == 1"})
    expect(r.score == 0.0, "code_exec fail")
    expect(extract_code("```python\nx=1\n```").strip() == "x=1", "extract fenced")

    print("checks.py selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
