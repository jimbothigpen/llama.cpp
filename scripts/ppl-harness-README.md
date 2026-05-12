# ppl-harness.py — PPL Regression Harness

Runs `llama-perplexity` against the ROCm and/or Vulkan installs of
llama-yggdrasil, writes structured JSON baselines to
`yggdrasil-context/ppl-baselines/`, and emits a PASS/WARN/FAIL parity
verdict between backends.

## Requirements

- Python 3.8+ (stdlib only — no venv needed)
- `llama-yggdrasil` built and installed to `/opt/llama-yggdrasil-{rocm,vulkan}/`
- wikitext-2 corpus (auto-fetched on first run)

## Quick start

```bash
cd /path/to/llama-yggdrasil

# Both backends, 20 chunks (~10k tokens)
./scripts/ppl-harness.py --model /path/to/model.gguf --chunks 20

# ROCm only, dry run (no baseline written)
./scripts/ppl-harness.py --model /path/to/model.gguf --backends rocm --no-record

# Vulkan only on ai01
./scripts/ppl-harness.py --model /path/to/model.gguf --backends vulkan

# Whole corpus (slow — omit --chunks or pass 0)
./scripts/ppl-harness.py --model /path/to/model.gguf --chunks 0
```

## Flags

| Flag | Default | Description |
|---|---|---|
| `--model PATH` | (required) | Path to `.gguf` model |
| `--backends LIST` | `rocm,vulkan` | Comma-separated backends; order = run order |
| `--wikitext PATH` | cephfs canonical | Path to `wiki.test.raw` |
| `--chunks N` | `0` (all) | Max chunks forwarded as `--chunks` to llama-perplexity |
| `--ctx-size N` | `512` | Context window size |
| `--ngl N` | `99` | GPU layers; use `0` for CPU-only |
| `--extra-flag FLAG` | (repeatable) | Verbatim passthrough to llama-perplexity |
| `--baseline-root PATH` | cephfs `ppl-baselines/` | Override baseline output directory |
| `--log-dir PATH` | `~/kernel-work/logs/session-*/` | Log directory |
| `--no-record` | off | Skip writing baseline JSON |
| `--no-sha256` | off | Skip sha256 (faster, less reproducible) |
| `--no-gpu-check` | off | Skip concurrent-process guard |
| `--wikitext-no-fetch` | off | Error if wikitext missing (don't auto-fetch) |
| `-v, --verbose` | off | Stream subprocess output to stderr |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | PASS — parity gate passed, or single backend |
| 1 | FAIL — parity delta ≥ 1.0% |
| 2 | A backend failed (subprocess error or PPL parse failure) |
| 3 | Invocation error (bad args, missing inputs) |

## Parity gate thresholds

| Delta | Verdict |
|---|---|
| < 0.5% | PASS |
| 0.5% – 1.0% | WARN (exit 0) |
| ≥ 1.0% | FAIL (exit 1) |

Delta = `abs(ppl_a - ppl_b) / min(ppl_a, ppl_b) * 100` (conservative).

## Baseline layout

```
yggdrasil-context/ppl-baselines/
└── <model-basename>/
    └── <git-commit>/             (prefix "dirty-" if tree is unclean)
        └── <host>__<backend>__<N>chunks.json
```

Each JSON file has schema version 1. Fields include: commit, branch,
host, model path + sha256, wikitext sha256, PPL value, parity gate
fields, wallclock time, and the exact CLI invocation.

## Mandatory runtime flags

`--no-mmap -fa on` are always passed to `llama-perplexity` — never omit.
This is baked into the harness and cannot be overridden from the CLI.

## Notes

- Backends run **sequentially** — llama-perplexity holds the GPU exclusively.
- The wikitext corpus is stored on cephfs at
  `yggdrasil-context/corpora/wikitext-2-raw/wiki.test.raw` so ai00 and
  ai01 read identical bytes (same sha256 = same slice = valid comparison).
- Model sha256 is cached at `~/.cache/yggdrasil/model-sha256.json`
  (keyed by path + mtime + size) to avoid rehashing large GGUFs.
- Do **not** use `ldconfig` system-wide for the yggdrasil library path —
  multiple forks live under `/opt/`. RPATH is embedded in the installed
  binaries (`$ORIGIN/../lib`).
