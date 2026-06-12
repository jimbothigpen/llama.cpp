from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, LazyTorchTensor, logger, gguf
from .llama import LlamaModel


@ModelBase.register("LlamaForCausalLMEagle3")
class LlamaEagle3Model(TextModel):
    model_arch = gguf.MODEL_ARCH.EAGLE3

    def _has_local_tokenizer(self) -> bool:
        return any(
            (self.dir_model / f).is_file()
            for f in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
        )

    def set_vocab(self) -> None:
        # SpecForge compact-vocab EAGLE3 drafts bundle no tokenizer; the GGUF must carry the
        # FULL target vocab (input embeddings + d2t scatter live in target-token space). Swap
        # dir_model to --target-model-dir to load the target's tokenizer (mirrors DFlashDraftModel).
        target_dir = getattr(self, "target_model_dir", None)
        needs_target = not self._has_local_tokenizer()

        if needs_target and target_dir is not None:
            logger.info("EAGLE3: no tokenizer in draft dir; loading tokenizer from --target-model-dir %s", target_dir)
            original_dir = self.dir_model
            self.dir_model = target_dir
        elif needs_target:
            raise ValueError(
                "EAGLE3: no tokenizer files in the draft model directory. Provide the base model "
                "path via --target-model-dir so the tokenizer can be loaded from there "
                "(e.g. --target-model-dir /path/to/Qwen3.5-35B-A3B)."
            )
        else:
            original_dir = None

        try:
            try:
                self._set_vocab_sentencepiece()
                return
            except FileNotFoundError:
                pass
            self._set_vocab_gpt2()
        finally:
            if original_dir is not None:
                self.dir_model = original_dir

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        eagle_config = self.hparams.get("eagle_config", {})
        extract_layers = eagle_config.get("eagle_aux_hidden_state_layer_ids", [1, 18, 35])
        self.gguf_writer.add_array(gguf.Keys.Eagle3.EXTRACT_LAYERS.format(arch="eagle3"), extract_layers)
        self.gguf_writer.add_uint32(gguf.Keys.Eagle3.TARGET_HIDDEN_SIZE.format(arch="eagle3"), self.hparams["hidden_size"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Skip t2d (target-to-draft, not needed for inference)
        if name == "t2d":
            return

        # d2t: draft-to-target vocab mapping. Compact-vocab drafts store it as an int offset table
        # (target_id = j + d2t[j]); keep it as raw I64 (written in prepare_tensors) so the parent
        # F32 quantize path does not mangle it.
        if name == "d2t":
            if not hasattr(self, "_eagle3_int_tensors"):
                self._eagle3_int_tensors = {}
            self._eagle3_int_tensors["d2t.weight"] = data_torch
            return

        # fc encoder projection
        if name == "fc.weight":
            yield ("fc.weight", data_torch)
            return

        # Per-aux fc_norm RMSNorm weights (fc_norm=True drafts). SpecForge/SGLang store one
        # RMSNorm per extracted aux layer as fcs.{i}.weight, applied to each target-feature
        # segment before the fc combiner (chunk → per-aux RMSNorm → concat → fc). We stash the
        # per-index [n_embd] vectors and emit them in prepare_tensors() as ONE packed tensor
        # fc_norm.weight of shape (n_aux, n_embd). A single (un-indexed) input-level tensor is
        # required: the loader rejects an input/output-class tensor that carries a layer index
        # (llama-model-loader.cpp create_tensor sanity check), so per-index fc_norm.{i} names
        # cannot be loaded as input tensors. The EAGLE3 driver slices row i per aux segment.
        if name.startswith("fcs.") and name.endswith(".weight") and name[4:-7].isdigit():
            idx = int(name[4:-7])
            if not hasattr(self, "_eagle3_fc_norm"):
                self._eagle3_fc_norm = {}
            self._eagle3_fc_norm[idx] = data_torch
            return

        # layers.N.xxx — alternative naming used by some EAGLE3 variants; normalize to midlayer.xxx.
        if name.startswith("layers."):
            parts = name.split(".", 2)
            if len(parts) == 3 and parts[1].isdigit():
                name = "midlayer." + parts[2]

        # Token embeddings
        if name == "embed_tokens.weight":
            yield ("token_embd.weight", data_torch)
            return

        # Output head
        if name == "lm_head.weight":
            yield ("output.weight", data_torch)
            return

        # Output norm
        if name == "norm.weight":
            yield ("output_norm.weight", data_torch)
            return

        # Midlayer (decoder transformer) tensors
        if name.startswith("midlayer."):
            midname = name[len("midlayer."):]

            # hidden_norm for g_embeddings normalization
            if midname == "hidden_norm.weight":
                yield ("blk.0.hidden_norm.weight", data_torch)
                return

            # Pre-attention norm (input_layernorm)
            if midname == "input_layernorm.weight":
                yield ("blk.0.attn_norm.weight", data_torch)
                return

            # Attention projections — note: input dim is 2*hidden (5120 for Bonsai-4B)
            if midname == "self_attn.q_proj.weight":
                n_head = self.hparams["num_attention_heads"]
                data_torch = LlamaModel.permute(data_torch, n_head, n_head)
                yield ("blk.0.attn_q.weight", data_torch)
                return
            if midname == "self_attn.k_proj.weight":
                n_head = self.hparams["num_attention_heads"]
                n_kv_head = self.hparams.get("num_key_value_heads", n_head)
                data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)
                yield ("blk.0.attn_k.weight", data_torch)
                return
            if midname == "self_attn.v_proj.weight":
                yield ("blk.0.attn_v.weight", data_torch)
                return
            if midname == "self_attn.o_proj.weight":
                yield ("blk.0.attn_output.weight", data_torch)
                return

            # Post-attention norm (FFN norm)
            if midname == "post_attention_layernorm.weight":
                yield ("blk.0.ffn_norm.weight", data_torch)
                return

            # MLP
            if midname == "mlp.gate_proj.weight":
                yield ("blk.0.ffn_gate.weight", data_torch)
                return
            if midname == "mlp.up_proj.weight":
                yield ("blk.0.ffn_up.weight", data_torch)
                return
            if midname == "mlp.down_proj.weight":
                yield ("blk.0.ffn_down.weight", data_torch)
                return

        raise ValueError(f"Unhandled EAGLE3 tensor: {name}")

    def prepare_tensors(self):
        super().prepare_tensors()

        # Write stashed integer tensors (d2t) as raw I64 — bypasses the F32 quantize path.
        # The lazy numpy() path has no int64 mapping, so materialize eagerly first.
        import torch
        for tensor_name, data_torch in getattr(self, "_eagle3_int_tensors", {}).items():
            data = LazyTorchTensor.to_eager(data_torch).to(torch.int64).numpy()
            shape_str = f"{{{', '.join(str(n) for n in reversed(data.shape))}}}"
            logger.info(f"{tensor_name + ',':<24} I64, shape = {shape_str}")
            self.gguf_writer.add_tensor(tensor_name, data, raw_dtype=gguf.GGMLQuantizationType.I64)

        # Pack per-aux fc_norm RMSNorm weights into ONE tensor fc_norm.weight of shape
        # (n_aux, n_embd) → ggml ne = [n_embd, n_aux], i.e. n_aux contiguous rows of n_embd.
        # F32 (norm weights stay full precision). Emitted only when the draft has fcs.* weights.
        fc_norm_map = getattr(self, "_eagle3_fc_norm", {})
        if fc_norm_map:
            rows = [LazyTorchTensor.to_eager(fc_norm_map[i]).to(torch.float32)
                    for i in sorted(fc_norm_map)]
            data = torch.stack(rows, dim=0).numpy()
            shape_str = f"{{{', '.join(str(n) for n in reversed(data.shape))}}}"
            logger.info(f"{'fc_norm.weight,':<24} F32, shape = {shape_str}")
            self.gguf_writer.add_tensor("fc_norm.weight", data, raw_dtype=gguf.GGMLQuantizationType.F32)
