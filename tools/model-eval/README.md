# Model-evaluation + tier-classification framework

A small, reusable harness that scores **any already-running OpenAI-compatible
chat server** on capability dimensions and assigns it a deployment **tier**.
Built for the local-serving fleet (122B-A10B / 35B-A3B / 27B-Coder / 9B-Coder /
4B-Coder): point it at a base URL + model id and it emits a scorecard you can
route on.

It only *queries* an endpoint — it never launches, restarts, or disrupts a
server — so it is safe to run against a live production server at modest volume.
(Contrast the sibling `tools/epicache-convqa` harness, which launches a server
per KV-policy arm.)

## Files

| file | role |
|---|---|
| `eval_harness.py` | driver: query server, run suite, score, classify tier, write JSON + markdown scorecard |
| `checks.py` | deterministic stdlib-only answer checks (EM / contains / any_of / all_of / regex / numeric / abstain / code_exec) — `--selftest` |
| `tiers.py` | dimension weights + tier threshold policy + classifier — `--selftest`, `--table` |
| `suites/default.json` | default `default-v1` probe: 20 items, 4 per dimension |
| `run_pilot.sh` | one-shot wrapper for a single model |
| `results/` | scorecards land here (`<model>.json` + `<model>.md`) |

## Capability dimensions

| dimension | what it probes | default check(s) |
|---|---|---|
| `code` | functional code generation | `code_exec` (run generated Python against asserts) |
| `reasoning` | grade-school math / logic | `numeric` (final-number extraction) |
| `instruction_following` | format / constraint adherence | `regex`, `all_of` |
| `knowledge` | short-form factual recall | `any_of` |
| `robustness` | abstention on unanswerable / adversarial prompts | `abstain` |

Each suite item declares its `dimension` and `check`; adding items or whole new
dimensions is data-only (edit the suite JSON). Per-dimension score = mean of its
items' scores.

## Tier policy

Composite = weight-normalized mean of the dimension scores. A model earns the
**highest** tier whose `min_composite` is met **and** whose per-dimension floors
all pass (floors stop a model buying a tier on easy dimensions while a critical
one — e.g. `code` for a coder deployment — is weak). Render live with
`python3 tiers.py --table`:

| tier | label | min composite | dimension floors | routing |
|---|---|---:|---|---|
| S | flagship | 0.85 | code≥0.80, reasoning≥0.75 | primary; eligible for hardest user traffic |
| A | production | 0.70 | code≥0.60, instruction_following≥0.60 | production-routable; general traffic |
| B | capable | 0.55 | code≥0.40 | fallback / overflow only |
| C | limited | 0.40 | — | non-critical / batch tasks; not user-facing |
| D | unreliable | — | — | do not serve; needs investigation |

Weights (default, coder-fleet tuned): code 0.30, reasoning 0.25,
instruction_following 0.20, knowledge 0.15, robustness 0.10. Tune in `tiers.py`.

## Quick start

```bash
cd tools/model-eval
python3 checks.py --selftest      # scorer sanity
python3 tiers.py  --selftest      # policy sanity

# score one served model (queries an existing server)
python3 eval_harness.py \
    --base-url http://127.0.0.1:8080 --model qwen36-35b \
    --suite suites/default.json --rps 0.7 \
    --out results/qwen36-35b.json --md results/qwen36-35b.md

# or the wrapper
MODEL=qwen36-35b BASE_URL=http://127.0.0.1:8080 ./run_pilot.sh
```

Per-deployment: bring each fleet model up on its own port and run the harness
against it — the scorecard's tier feeds the routing program. Nothing in the
harness is backend- or model-specific.

### Useful flags

- `--dimensions code,reasoning` — restrict to a subset of dimensions
- `--max-items N` — cap items (smoke test)
- `--rps R` — request rate cap (default 1.0; use ≤1 against prod)
- `--thinking` — allow reasoning-model chain-of-thought (default off: we
  suppress it via `chat_template_kwargs.enable_thinking=false` and read
  `reasoning_content` only as a fallback, so the final answer lands in `content`)
- `--no-code-exec` — disable running model-generated code (then `code_exec`
  items score 0; use when the harness host is untrusted)

## Reproducibility & safety

- Greedy decoding (`temperature=0, top_k=1`) and `cache_prompt:false` (every
  prompt re-prefills) → deterministic given the server build.
- Scoring is offline, stdlib-only — a run is reproducible from the results JSON.
- `code_exec` runs model-generated Python in a **subprocess with a wall-clock
  timeout and no stdin**. It is on by default for trusted local hosts; pass
  `--no-code-exec` otherwise.
- Stamp the server build SHA (`/v1/models` meta or `llama-server --version`)
  alongside results when comparing across model versions.

## Sample scorecard (35B pilot, 2026-06-20)

Interim **qwen36-35b @ Q5_K_M** (not the final IQ4_KS deployment) on the live
production server `http://127.0.0.1:8080`, suite `default-v1` (20 items):

| dimension | n | mean |
|---|---:|---:|
| code | 4 | 1.000 |
| reasoning | 4 | 1.000 |
| instruction_following | 4 | 1.000 |
| knowledge | 4 | 1.000 |
| robustness | 4 | 1.000 |
| **composite** | | **1.000** |

→ **TIER S (flagship)**. The interim 35B aces the `default-v1` probe; the suite
is intentionally small for a pilot and does not yet discriminate at the top end
(see follow-ups). Full scorecard: `results/qwen36-35b-pilot.md`.

## Follow-ups

- Run on each real deployment once the fleet is up (122B / 27B-Coder /
  9B-Coder / 4B-Coder), and re-run 35B on the **final IQ4_KS** build.
- Add harder items (multi-step code, harder math, longer-context retrieval,
  trickier adversarial abstention) so strong models spread out below 1.0 — the
  current suite saturates on the 35B.
- Optional: add a long-context / needle-in-haystack dimension (the server
  advertises `n_ctx_train=1048576`).
- Optional LLM-judge layer over the results JSON for open-ended items; the
  deterministic checks remain the reproducible default.
