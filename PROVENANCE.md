# Provenance

Credit and upstream source for features ported into this fork from external contributors.
One row per ported feature. See `docs/features/` for per-feature detail and `docs/TYPE_ASSIGNMENTS.md`
for ggml type-slot lineage.

| Feature | ggml type / slot | Upstream source | Lineage credit | Doc |
|---|---|---|---|---|
| WQ3 TCQ (3-bit trellis-coded weight quant + FFN fusion) | `GGML_TYPE_WQ3_TCQ` = 92 | buun `feat/tcq-wq3-ffn-fusion` (`704cc7780`, `77355efa2`, `d0e929b0e`) | TCQ algorithm: TheTom (TurboQuant TCQ) | [`docs/features/wq3-tcq.md`](docs/features/wq3-tcq.md) |
| WQ3 TCQ quantizer (CPU Viterbi encoder + CPU dequant + codebook + imatrix decision) | `LLAMA_FTYPE_MOSTLY_WQ3_TCQ` = 59 | encoder ported from buun `scripts/tcq_rshift.py` + `scripts/analyze_norm_correction.py` (right-shift trellis, corrected-norm); k=3/L=10 to match the runtime decode | TCQ algorithm: TheTom (TurboQuant TCQ) | [`docs/features/wq3-tcq.md`](docs/features/wq3-tcq.md) |
