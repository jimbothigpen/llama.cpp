# Provenance

Credit and upstream source for features ported into this fork from external contributors.
One row per ported feature. See `docs/features/` for per-feature detail and `docs/TYPE_ASSIGNMENTS.md`
for ggml type-slot lineage.

| Feature | ggml type / slot | Upstream source | Lineage credit | Doc |
|---|---|---|---|---|
| WQ3 TCQ (3-bit trellis-coded weight quant + FFN fusion) | `GGML_TYPE_WQ3_TCQ` = 92 | buun `feat/tcq-wq3-ffn-fusion` (`704cc7780`, `77355efa2`, `d0e929b0e`) | TCQ algorithm: TheTom (TurboQuant TCQ) | [`docs/features/wq3-tcq.md`](docs/features/wq3-tcq.md) |
