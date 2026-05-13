# GGML op-enum assignments

Yggdrasil-original GGML ops have no explicit integer assignments. The mainline
`ggml/include/ggml.h` `enum ggml_op` enumerates positionally (no explicit
values) and ends in `GGML_OP_COUNT`. New ops append immediately before
`GGML_OP_COUNT`.

Per Phase 0.5 constraint **C1.3**, yggdrasil-added ops are listed here so the
intent and ordering are explicit and audit-able. The "slot" column is the
position-index in `enum ggml_op` after each op is inserted (counting from 0).
Mainline-side reordering will shift slot numbers — the contract is
**positional, not numeric**.

| Op | First landed | Phase | Purpose |
|---|---|---|---|
| `GGML_OP_TURBO_WHT` | feature/phase-1-turboquant-kv Step 3 | Phase 1 | TurboQuant Walsh-Hadamard Transform: O(d log d) rotation for KV-cache compression. Replaces dense `ggml_mul_mat(128x128, ...)`. Sourced from TheTom `012faec26` (introducing commit). |

## Update protocol

When adding a yggdrasil-original `GGML_OP_*`:

1. Append the op to `enum ggml_op` immediately before `GGML_OP_COUNT` in
   `ggml/include/ggml.h`.
2. Append a matching entry to `GGML_OP_NAME[]` and `GGML_OP_SYMBOL[]` in
   `ggml/src/ggml.c` (and bump both `static_assert(GGML_OP_COUNT == N, ...)`
   counters).
3. Wire CPU dispatch in `ggml/src/ggml-cpu/ggml-cpu.c` `ggml_compute_forward`
   and `ggml_get_n_tasks` and `ggml_graph_plan`.
4. Add CPU impl in `ggml/src/ggml-cpu/ops.cpp` + declaration in `ops.h`.
5. Add backend supports_op gates: ROCm (`ggml-cuda.cu`), Vulkan
   (`ggml-vulkan.cpp`), CPU (auto via dispatch).
6. Append a row to this table with the first-landed branch/phase and a
   one-sentence purpose.
