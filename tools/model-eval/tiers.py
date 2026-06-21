#!/usr/bin/env python3
"""Capability-dimension weighting + tier classification policy.

The policy is data (``POLICY``) so it is tunable per deployment fleet and can be
rendered as a threshold table for docs / briefs.  Two pieces:

  WEIGHTS         per-dimension weight for the composite (weighted mean over the
                  dimensions actually present in a run; missing dimensions are
                  dropped and the remaining weights renormalized, so a partial
                  suite still yields a meaningful composite).

  TIERS           ordered best->worst.  A model earns the *highest* tier whose
                  ``min_composite`` is met AND whose per-dimension ``floors`` are
                  all satisfied.  Floors stop a model from buying a high tier on
                  the strength of easy dimensions while a critical one (e.g.
                  ``code`` for a coder deployment) is weak.

Tiers map to the local-serving routing program: S/A are production-routable,
B is fallback-only, C/D should not serve user traffic.
"""
from __future__ import annotations

# Dimension weights for the composite score.  Tuned for the coder-heavy fleet
# (122B/35B/27B-Coder/9B-Coder/4B-Coder): code + reasoning dominate.
WEIGHTS = {
    "code": 0.30,
    "reasoning": 0.25,
    "instruction_following": 0.20,
    "knowledge": 0.15,
    "robustness": 0.10,
}

# Ordered best -> worst. floors: per-dimension minimum to qualify for the tier.
TIERS = [
    {"tier": "S", "label": "flagship",   "min_composite": 0.85,
     "floors": {"code": 0.80, "reasoning": 0.75},
     "routing": "primary; eligible for hardest user traffic"},
    {"tier": "A", "label": "production", "min_composite": 0.70,
     "floors": {"code": 0.60, "instruction_following": 0.60},
     "routing": "production-routable; general traffic"},
    {"tier": "B", "label": "capable",    "min_composite": 0.55,
     "floors": {"code": 0.40},
     "routing": "fallback / overflow only"},
    {"tier": "C", "label": "limited",    "min_composite": 0.40,
     "floors": {},
     "routing": "non-critical / batch tasks; not user-facing"},
    {"tier": "D", "label": "unreliable", "min_composite": 0.0,
     "floors": {},
     "routing": "do not serve; needs investigation"},
]

POLICY = {"weights": WEIGHTS, "tiers": TIERS, "version": "tier-policy-v1"}


def composite(dim_scores: dict) -> float:
    """Weighted mean over present dimensions, renormalizing dropped weights."""
    used = {d: s for d, s in dim_scores.items() if d in WEIGHTS}
    wsum = sum(WEIGHTS[d] for d in used)
    if wsum <= 0:
        return 0.0
    return sum(WEIGHTS[d] * s for d, s in used.items()) / wsum


def classify(dim_scores: dict):
    """Return (tier_record, comp, reasons) for the highest tier the model earns.

    ``reasons`` explains why higher tiers were skipped (composite short or a
    floor failed), which the scorecard surfaces so a tier is never opaque.
    """
    comp = composite(dim_scores)
    reasons = []
    for t in TIERS:
        if comp < t["min_composite"]:
            reasons.append(f"{t['tier']}: composite {comp:.3f} < {t['min_composite']:.2f}")
            continue
        failed = [
            f"{d}={dim_scores.get(d, 0.0):.3f}<{floor:.2f}"
            for d, floor in t["floors"].items()
            if dim_scores.get(d, 0.0) < floor
        ]
        if failed:
            reasons.append(f"{t['tier']}: floor fail [{', '.join(failed)}]")
            continue
        return t, comp, reasons
    return TIERS[-1], comp, reasons


def threshold_table_md() -> str:
    """Render the tier policy as a markdown table (for README / brief)."""
    lines = [
        f"Composite = weighted mean of dimension scores (weights: "
        + ", ".join(f"{d} {w:.2f}" for d, w in WEIGHTS.items()) + ").",
        "",
        "| tier | label | min composite | dimension floors | routing |",
        "|---|---|---:|---|---|",
    ]
    for t in TIERS:
        floors = ", ".join(f"{d}≥{v:.2f}" for d, v in t["floors"].items()) or "—"
        mc = f"{t['min_composite']:.2f}" if t["min_composite"] > 0 else "—"
        lines.append(f"| {t['tier']} | {t['label']} | {mc} | {floors} | {t['routing']} |")
    return "\n".join(lines)


def _selftest() -> int:
    ok = True

    def expect(cond, msg):
        nonlocal ok
        if not cond:
            print("FAIL:", msg)
            ok = False

    # Strong all-round -> S
    s = {"code": 0.9, "reasoning": 0.85, "instruction_following": 0.9,
         "knowledge": 0.8, "robustness": 0.9}
    t, c, _ = classify(s)
    expect(t["tier"] == "S", f"strong->S got {t['tier']} comp={c:.3f}")

    # High composite but weak code -> demoted below S (code floor 0.80)
    s2 = {"code": 0.5, "reasoning": 0.95, "instruction_following": 0.95,
          "knowledge": 0.95, "robustness": 0.95}
    t2, c2, _ = classify(s2)
    expect(t2["tier"] != "S", f"weak-code blocked from S got {t2['tier']}")

    # Mid -> B or C
    s3 = {"code": 0.45, "reasoning": 0.5, "instruction_following": 0.5,
          "knowledge": 0.6, "robustness": 0.5}
    t3, c3, _ = classify(s3)
    expect(t3["tier"] in ("B", "C"), f"mid got {t3['tier']} comp={c3:.3f}")

    # Partial suite (only code+reasoning) still classifies
    t4, c4, _ = classify({"code": 0.9, "reasoning": 0.9})
    expect(0.0 < c4 <= 1.0, f"partial composite {c4}")

    print("tiers.py selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    if "--table" in sys.argv:
        print(threshold_table_md())
    else:
        print(__doc__)
