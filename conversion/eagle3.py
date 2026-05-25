from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf
from .llama import LlamaModel


@ModelBase.register("LlamaForCausalLMEagle3")
class LlamaEagle3Model(TextModel):
    model_arch = gguf.MODEL_ARCH.EAGLE3

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

        # d2t: draft-to-target vocab mapping
        if name == "d2t":
            yield ("d2t.weight", data_torch)
            return

        # fc encoder projection
        if name == "fc.weight":
            yield ("fc.weight", data_torch)
            return

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
