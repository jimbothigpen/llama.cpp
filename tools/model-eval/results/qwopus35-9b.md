# Model eval scorecard — qwopus35-9b

- endpoint: `http://127.0.0.1:8184`
- suite: `default-v1` (20 items)
- server: n_params=8953803264 n_ctx_train=262144 size=5061101568
- policy: `tier-policy-v1`

## TIER: **B** (capable) — composite **0.812**

> routing: fallback / overflow only

### Per-dimension

| dimension | n | mean score |
|---|---:|---:|
| code | 4 | 1.000 |
| instruction_following | 4 | 0.500 |
| knowledge | 4 | 1.000 |
| reasoning | 4 | 0.750 |
| robustness | 4 | 0.750 |
| **composite** |  | **0.812** |

### Why not a higher tier

- S: composite 0.812 < 0.85
- A: floor fail [instruction_following=0.500<0.60]

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
| reason-2 | reasoning | 0.00 | got=54.0 want=39.0 |
| if-1 | instruction_following | 0.00 |  |
| if-2 | instruction_following | 0.00 |  |
| robust-3 | robustness | 0.00 | answered |
