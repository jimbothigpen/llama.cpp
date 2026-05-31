# Ported from Anbeeld/beellama.cpp (MIT) — TODO 122
from __future__ import annotations

from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, logger, gguf


@ModelBase.register("DFlashDraftModel")
class DFlashDraftModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DFLASH_DRAFT

    def _is_gemma4_dflash(self) -> bool:
        hints = [
            str(self.hparams.get("model_type", "")),
            str(self.hparams.get("_name_or_path", "")),
            str(self.hparams.get("name_or_path", "")),
            str(self.dir_model.name),
        ]
        archs = self.hparams.get("architectures", [])
        if isinstance(archs, list):
            hints.extend(str(x) for x in archs)

        hints_l = [h.lower() for h in hints if h]
        return any(("gemma" in h and "4" in h) or "gemma-4" in h for h in hints_l)

    def _set_vocab_gemma4_hf_bpe(self) -> None:
        vocab = gguf.LlamaHfVocab(self.dir_model)
        tokens = []
        scores = []
        toktypes = []

        visible_tokens = {
            b"<|channel>",
            b"<channel|>",
            b"<|tool_call>",
            b"<tool_call|>",
            b"<|tool_response>",
            b"<tool_response|>",
            b'<|"|>',
        }

        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)

            if isinstance(text, str):
                text_bytes = text.encode("utf-8")
            elif isinstance(text, memoryview):
                text_bytes = text.tobytes()
            else:
                text_bytes = bytes(text)

            if text_bytes in visible_tokens:
                toktypes.append(gguf.TokenType.USER_DEFINED)
                logger.info(
                    "Token %r is set to USER_DEFINED",
                    text_bytes.decode("utf-8", errors="replace"),
                )
            else:
                toktypes.append(toktype)

        assert len(tokens) == vocab.vocab_size

        self.gguf_writer.add_tokenizer_model("gemma4")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)
        self.gguf_writer.add_add_space_prefix(False)
        self.gguf_writer.add_add_bos_token(True)

    def _has_local_tokenizer(self) -> bool:
        return any(
            (self.dir_model / f).exists()
            for f in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
        )

    def set_vocab(self) -> None:
        # Swap dir_model temporarily when the draft has no bundled tokenizer and
        # --target-model-dir points to the base model that holds the tokenizer.
        target_dir = getattr(self, "target_model_dir", None)
        needs_target = not self._has_local_tokenizer()

        if needs_target and target_dir is not None:
            logger.info(
                "DFlashDraftModel: no tokenizer in draft dir; loading tokenizer from --target-model-dir %s",
                target_dir,
            )
            original_dir = self.dir_model
            self.dir_model = target_dir
        elif needs_target:
            raise ValueError(
                "DFlashDraftModel: no tokenizer files found in the draft model directory. "
                "Provide the base model path via --target-model-dir so the tokenizer can be "
                "loaded from there (e.g. --target-model-dir /path/to/Qwen3.5-27B)."
            )
        else:
            original_dir = None

        try:
            try:
                self._set_vocab_sentencepiece()
                return
            except FileNotFoundError:
                pass

            if self._is_gemma4_dflash():
                try:
                    self._set_vocab_gemma4_hf_bpe()
                    return
                except (FileNotFoundError, TypeError, ValueError, UnicodeDecodeError) as e:
                    logger.warning(
                        "DFlashDraftModel: Gemma4 HF/BPE vocab path failed: %s; falling back to GPT-2 vocab",
                        e,
                    )

            self._set_vocab_gpt2()
        finally:
            if original_dir is not None:
                self.dir_model = original_dir

    def set_gguf_parameters(self) -> None:
        super().set_gguf_parameters()

        self.gguf_writer.add_causal_attention(False)

        head_dim = self.hparams.get("head_dim", 128)
        self.gguf_writer.add_rope_dimension_count(head_dim)

        arch = self.gguf_writer.arch

        # newer drafters nest dflash-specific fields under "dflash_config"; fall back to top-level for older ones
        dflash_cfg = self.hparams.get("dflash_config", {})

        def dflash_value(name: str, default: Any) -> Any:
            if name in dflash_cfg:
                return dflash_cfg[name]
            if name in self.hparams:
                return self.hparams[name]
            logger.warning("DFlashDraftModel: missing %s; using default %r", name, default)
            return default

        block_size = dflash_value("block_size", 16)
        self.gguf_writer.add_uint32(f"{arch}.dflash.block_size", block_size)

        mask_token_id = dflash_value("mask_token_id", 248070)
        self.gguf_writer.add_uint32(f"{arch}.dflash.mask_token_id", mask_token_id)

        target_layer_ids = dflash_value("target_layer_ids", [1, 16, 31, 46, 61])
        self.gguf_writer.add_array(f"{arch}.dflash.target_layer_ids", target_layer_ids)

        if "n_target_features" in dflash_cfg:
            n_target_features = dflash_cfg["n_target_features"]
        elif "n_target_features" in self.hparams:
            n_target_features = self.hparams["n_target_features"]
        else:
            n_target_features = self.hparams.get("hidden_size", 5120) * len(target_layer_ids)
            logger.warning(
                "DFlashDraftModel: missing n_target_features; inferred %d = hidden_size(%d) * n_target_layers(%d)",
                n_target_features,
                self.hparams.get("hidden_size", 5120),
                len(target_layer_ids),
            )

        self.gguf_writer.add_uint32(f"{arch}.dflash.n_target_features", n_target_features)

        logger.info(
            "DFlashDraftModel metadata: block_size=%s mask_token_id=%s target_layer_ids=%s n_target_features=%s",
            block_size,
            mask_token_id,
            target_layer_ids,
            n_target_features,
        )

        if self.hparams.get("use_sliding_window") and self.hparams.get("sliding_window"):
            self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
            layer_types = self.hparams.get("layer_types", [])
            pattern = [t == "sliding_attention" for t in layer_types]
            if pattern:
                self.gguf_writer.add_sliding_window_pattern(pattern)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # strip "model." prefix if present
        if name.startswith("model."):
            name = name[len("model."):]

        # DFlash-specific global tensors not in the standard HF name map
        if name == "fc.weight":
            yield ("dflash_fc.weight", data_torch)
            return
        if name == "hidden_norm.weight":
            yield ("dflash_hidden_norm.weight", data_torch)
            return

        # base class maps post_attention_layernorm → ffn_norm but DFlash loader
        # requires it as attn_post_norm (blk.X.post_attention_norm)
        if bid is not None and "post_attention_layernorm" in name:
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_POST_NORM, bid=bid), data_torch)
            return

        yield from super().modify_tensors(data_torch, name, bid)
