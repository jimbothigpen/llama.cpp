# Primer: Feature Maturity Levels & Backend Support

Each feature doc includes a **Status banner** at the top. This primer defines what those labels mean and how backend support is documented.

## Maturity levels

| Label | Meaning |
|---|---|
| **Stable** | Shipped in production builds. CPU, primary GPU backends (CUDA/HIP and Vulkan) all pass cross-backend parity checks. API and CLI flags are stable. |
| **Experimental — known limitation: …** | Functional but with a documented caveat (e.g., "Vulkan encode path missing — falls back to CPU"). May have known accuracy regressions on some architectures. API may change. |
| **Preview — not in released builds** | Code is merged to the development branch but not yet included in a tagged release build. Behavior may change before stabilization. |

## Backend support notation

Feature docs list backend support as a comma-separated list in the at-a-glance table:

| Symbol | Backend |
|---|---|
| CPU | GGML CPU backend (all platforms) |
| CUDA | NVIDIA CUDA (via `ggml-cuda`) |
| HIP | AMD ROCm/HIP (compiled from the same `ggml-cuda` source with `-DGGML_HIP=ON`) |
| Vulkan | Vulkan compute (via `ggml-vulkan`) |
| Metal | Apple Metal (via `ggml-metal`; when listed) |

When CUDA and HIP share the same implementation path (the common case for kernels in `ggml/src/ggml-cuda/`), they are written together as **CUDA/HIP**.

## What "CPU fallback" means

For some types or operations, GPU backends lack a native kernel. In that case the runtime transparently moves the affected tensor to a CPU buffer so the CPU backend handles it. Performance is degraded but correctness is preserved. Feature docs note "CPU fallback" when this applies.

## Further reading

- [docs/features/README.md](../README.md) — index of all feature docs
- [docs/BACKEND_PARITY.md](../../BACKEND_PARITY.md) — parity test methodology and results
