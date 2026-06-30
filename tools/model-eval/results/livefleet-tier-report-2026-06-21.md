# Live-fleet tier-classification report — 2026-06-21

Harness: `model-eval-framework` / suite `default-v1` (20 items, 4 per dimension)  
Policy: `tier-policy-v1` (code×0.30, reasoning×0.25, instruction_following×0.20, knowledge×0.15, robustness×0.10)  
Run date: 2026-06-21  
Status: **COMPLETE** — all 5 fleet models evaluated (122B + 35B-final added 2026-06-21)

---

## Fleet ranking

| rank | model | quant | n_ctx | composite | **tier** | routing |
|---:|---|---|---:|---:|---|---|
| 1 | Qwen3.6-35B-A3B **(final)** | IQ4_KS | 786 432 | **1.000** | **S** flagship | primary |
| 2 | Qwen3.5-122B-A10B | — | 1 048 576 | **0.975** | **S** flagship | primary |
| 2 | Qwen3.6-27B-Coder | IQ4_KS | 262 144 | **0.975** | **S** flagship | primary |
| 4 | Qwen3.5-4B-Coder | IQ4_KS | 393 216 | **0.912** | **S** flagship ⚠️ | primary *(eval only — see caveat)* |
| 5 | Qwen3.5-9B-Coder | IQ4_KS | 262 144 | **0.812** | **B** capable | fallback / overflow only |

> **⚠️ Agentic-routing caveat (122B, 27B):** Discrete eval is not sufficient proof of agentic-worker viability — the 4B proved this (Tier S eval score, worker FAIL). A dedicated agentic task battery must be run before routing 122B or 27B as orchestrator/spawned-worker.

---

## Per-model scorecards

### 1. Qwen3.6-35B-A3B (final IQ4_KS) — **S** (composite 1.000)

> **Final production deployment** (`qwen36-35b-iq4ks`, 768K ctx). Supersedes the interim Q5_K_M pilot. Perfect score across all 5 dimensions — only model on the fleet to pass robust-3. Confirms that IQ4_KS quantization introduces no degradation vs the pilot Q5_K_M (both 1.000). This is the strongest all-around model on the fleet at default-v1 granularity.

| dimension | n | score |
|---|---:|---:|
| code | 4 | 1.000 |
| reasoning | 4 | 1.000 |
| instruction_following | 4 | 1.000 |
| knowledge | 4 | 1.000 |
| robustness | 4 | 1.000 |
| **composite** | | **1.000** |

Failed items: none

---

### 2. Qwen3.5-122B-A10B — **S** (composite 0.975)

> Flagship MoE (10B-active parameters, 1M context). Near-perfect; single failure on robust-3 (answered when should abstain) — identical failure mode to the 27B. Tier S holds comfortably above composite floor (0.975 >> 0.85) and all dimension floors. Extended-context (1M) advantage makes this the preferred routing target for long-horizon tasks once agentic battery confirms worker viability.

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

### 3. Qwen3.6-27B-Coder — **S** (composite 0.975)

> Near-perfect across all dimensions. Single failure on robust-3 (answered when it should abstain). Tier S holds comfortably above composite floor (0.975 >> 0.85) and all dimension floors. Primary routing cleared for coder traffic. Tied with 122B at composite; ranked below on extended-context capability.

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

### 4. Qwen3.5-9B-Coder — **B** (composite 0.812)

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

### 5. Qwen3.5-4B-Coder — **S** ⚠️ eval score, **NOT-viable agentic worker**

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

1. **35B IQ4_KS final matches pilot** (both 1.000) — no quant degradation. The final deployment is the strongest all-around model on the fleet, uniquely passing robust-3 across all runs.

2. **122B and 27B tied at Tier S / 0.975** — both fail only robust-3. The 122B 1M context makes it preferred for long-horizon task routing. Both need agentic battery before worker deployment (see finding 5).

3. **robust-3 is the consistent discriminator** — 4 of 5 models fail it (answer when should abstain). Only the 35B IQ4_KS passes. This item appears to be a meaningful robustness signal; consider extending the suite with similar abstain probes.

4. **9B instruction_following weakness** (0.5) is the blocking issue for A-tier. Items `if-1` and `if-2` — the harness raw JSON has the exact prompts for targeted debugging. The 9B should not handle user-facing routing until this is resolved.

5. **Agentic battery required for 122B/27B** — the 4B agentic failure demonstrated that discrete eval scores do not predict worker viability. Before routing 122B or 27B as agentic workers, run a dedicated multi-step agentic task battery. This is the single most important follow-up for the fleet.

6. **35B pilot superseded** — `results/qwen36-35b-pilot.{json,md}` are the interim Q5_K_M results; `results/qwen36-35b-iq4ks.{json,md}` are canonical for the live deployment.

---

## Scorecard files

| model | JSON | markdown | note |
|---|---|---|---|
| qwen36-35b-iq4ks | `results/qwen36-35b-iq4ks.json` | `results/qwen36-35b-iq4ks.md` | final; supersedes pilot |
| qwen35-122b | `results/qwen35-122b.json` | `results/qwen35-122b.md` | flagship MoE |
| qwopus36-27b | `results/qwopus36-27b.json` | `results/qwopus36-27b.md` | |
| qwopus35-9b | `results/qwopus35-9b.json` | `results/qwopus35-9b.md` | |
| qwopus35-4b | `results/qwopus35-4b.json` | `results/qwopus35-4b.md` | eval-class S / worker-class FAIL |
| qwen36-35b-pilot | `results/qwen36-35b-pilot.json` | `results/qwen36-35b-pilot.md` | *(interim Q5_K_M — superseded)* |
