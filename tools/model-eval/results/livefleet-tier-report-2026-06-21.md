# Live-fleet tier-classification report — 2026-06-21

Harness: `model-eval-framework` / suite `default-v1` (20 items, 4 per dimension)  
Policy: `tier-policy-v1` (code×0.30, reasoning×0.25, instruction_following×0.20, knowledge×0.15, robustness×0.10)  
Run date: 2026-06-21  
Status: **PARTIAL** — 4 of 5 fleet models evaluated; 122B pending (ai00 standup gated, user-scheduled)

---

## Fleet ranking

| rank | model | quant | n_ctx | composite | **tier** | routing |
|---:|---|---|---:|---:|---|---|
| 1 | Qwopus3.6-35B (pilot) | Q5_K_M *(interim)* | 1 048 576 | **1.000** | **S** flagship | primary |
| 2 | Qwopus3.6-27B-Coder | IQ4_KS | 262 144 | **0.975** | **S** flagship | primary |
| 3 | Qwopus3.5-4B-Coder | IQ4_KS | 393 216 | **0.912** | **S** flagship ⚠️ | primary *(eval only — see caveat)* |
| 4 | Qwopus3.5-9B-Coder | IQ4_KS | 262 144 | **0.812** | **B** capable | fallback / overflow only |
| — | Qwopus3.6-122B | — | — | *pending* | *TBD* | *TBD* |

---

## Per-model scorecards

### 1. Qwopus3.6-35B pilot — **S** (composite 1.000)

> **Interim Q5_K_M** — not yet the final IQ4_KS deployment. Saturates the default-v1 suite on all 5 dimensions. Serves as the performance ceiling reference. IQ4_KS final quant expected to match or come within 1–2 pp.

| dimension | n | score |
|---|---:|---:|
| code | 4 | 1.000 |
| reasoning | 4 | 1.000 |
| instruction_following | 4 | 1.000 |
| knowledge | 4 | 1.000 |
| robustness | 4 | 1.000 |
| **composite** | | **1.000** |

---

### 2. Qwopus3.6-27B-Coder — **S** (composite 0.975)

> Near-perfect across all dimensions. Single failure on robust-3 (answered when it should abstain). Tier S holds comfortably above composite floor (0.975 >> 0.85) and all dimension floors. Primary routing cleared.

| dimension | n | score |
|---|---:|---:|
| code | 4 | 1.000 |
| reasoning | 4 | 1.000 |
| instruction_following | 4 | 1.000 |
| knowledge | 4 | 1.000 |
| robustness | 4 | **0.750** |
| **composite** | | **0.975** |

Failed items: `robust-3` (robustness, score 0 — answered when abstain expected)

---

### 3. Qwopus3.5-9B-Coder — **B** (composite 0.812)

> ⚠️ **Fails Tier A floor** on `instruction_following` (0.50 < required 0.60). Failed 2/4 instruction-following items (`if-1`, `if-2`) and 1/4 reasoning item (`reason-2`, arithmetic error: got 54 want 39). Cannot be promoted to A without addressing these weaknesses. Capped at Tier B = fallback/overflow only — **not user-facing, not primary routing.**

| dimension | n | score |
|---|---:|---:|
| code | 4 | **1.000** |
| reasoning | 4 | 0.750 |
| instruction_following | 4 | **0.500** ← A-floor fail |
| knowledge | 4 | **1.000** |
| robustness | 4 | 0.750 |
| **composite** | | **0.812** |

Why not higher:
- S: composite 0.812 < 0.85
- A: floor fail [instruction_following=0.500 < 0.60]

Failed items: `reason-2` (got 54, want 39), `if-1` and `if-2` (instruction_following, both score 0), `robust-3` (answered)

---

### 4. Qwopus3.5-4B-Coder — **S** ⚠️ eval score, **NOT-viable agentic worker**

> **Critical caveat — eval score does NOT reflect real-world agentic capability.** This model scores Tier S on the default-v1 suite (composite 0.912, all short-form probes). However, it **failed a production agentic `--print` worker task**: ran ~57 minutes, produced no deliverable, and exited 1. The eval suite probes discrete correctness on short-horizon tasks; it does not capture multi-step planning, instruction persistence over long contexts, or recovery from partial failure — all required for worker-tier deployment.
>
> **Routing recommendation:** Do NOT use as an agentic worker (orchestrator, spawned worker, long-horizon planner). May serve simple short-form completions, autocomplete, or other stateless single-turn tasks where its actual capability aligns with the eval. Flag this model as **eval-class S / worker-class FAIL** until a dedicated agentic task battery is run.

| dimension | n | score |
|---|---:|---:|
| code | 4 | 1.000 |
| reasoning | 4 | **0.750** |
| instruction_following | 4 | 1.000 |
| knowledge | 4 | 1.000 |
| robustness | 4 | **0.750** |
| **composite** | | **0.912** |

Failed items: `reason-3` (got 18, want 8), `robust-3` (answered)

---

## Key findings

1. **27B is the strongest coder on the fleet** (0.975, Tier S) with only one robustness failure. Recommended as primary routing target for coder traffic.

2. **9B instruction_following weakness** (0.5) is the blocking issue for A-tier. Items `if-1` and `if-2` — the harness raw JSON has the exact prompts for targeted debugging. The 9B should not handle user-facing routing until this is resolved.

3. **4B eval-vs-real-world gap** is the most significant finding of this run. A Tier S eval score on discrete probes is misleading for models that cannot maintain coherent multi-step plans over long horizons. The 4B agentic failure is documented and must be propagated to any routing policy that uses this report.

4. **35B pilot saturates default-v1** — once the final IQ4_KS 35B is deployed, re-run with an extended suite (long-context needle, multi-turn, agentic mini-tasks) to differentiate the top-tier models.

5. **122B pending** — expected Tier S+ with extended-context advantages; add to the report when the ai00 standup completes.

---

## Scorecard files

| model | JSON | markdown |
|---|---|---|
| qwopus36-35b (pilot) | `results/qwen36-35b-pilot.json` | `results/qwen36-35b-pilot.md` |
| qwopus36-27b | `results/qwopus36-27b.json` | `results/qwopus36-27b.md` |
| qwopus35-9b | `results/qwopus35-9b.json` | `results/qwopus35-9b.md` |
| qwopus35-4b | `results/qwopus35-4b.json` | `results/qwopus35-4b.md` |
