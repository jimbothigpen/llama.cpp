# Model eval scorecard — qwopus36-27b

- endpoint: `http://127.0.0.1:8080`
- suite: `default-v1` (20 items)
- server: n_params=26895998464 n_ctx_train=262144 size=14681798656
- policy: `tier-policy-v1`

## TIER: **S** (flagship) — composite **0.975**

> routing: primary; eligible for hardest user traffic

### Per-dimension

| dimension | n | mean score |
|---|---:|---:|
| code | 4 | 1.000 |
| instruction_following | 4 | 1.000 |
| knowledge | 4 | 1.000 |
| reasoning | 4 | 1.000 |
| robustness | 4 | 0.750 |
| **composite** |  | **0.975** |

### Tier policy

Composite = weighted mean of dimension scores (weights: code 0.30, reasoning 0.25, instruction_following 0.20, knowledge 0.15, robustness 0.10).

| tier | label | min composite | dimension floors | routing |
|---|---|---:|---|---|
| S | flagship | 0.85 | code≥0.80, reasoning≥0.75 | primary; eligible for hardest user traffic |
| A | production | 0.70 | code≥0.60, instruction_following≥0.60 | production-routable; general traffic |
| B | capable | 0.55 | code≥0.40 | fallback / overflow only |
| C | limited | 0.40 | — | non-critical / batch tasks; not user-facing |
| D | unreliable | — | — | do not serve; needs investigation |

### Failed / notable items

| id | dim | score | detail |
|---|---|---:|---|
| robust-3 | robustness | 0.00 | answered |
