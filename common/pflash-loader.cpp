#include "pflash-loader.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static ggml_tensor * get_tensor(ggml_context * ctx, const char * name) {
	ggml_tensor * t = ggml_get_tensor(ctx, name);
	if (!t) {
		fprintf(stderr, "pflash: missing tensor '%s'\n", name);
	}
	return t;
}

int pflash_model_load(pflash_model & model, const std::string & gguf_path, int gpu_device) {
	// mmap the GGUF file
	model.mmap_fd = open(gguf_path.c_str(), O_RDONLY);
	if (model.mmap_fd < 0) {
		fprintf(stderr, "pflash: cannot open '%s'\n", gguf_path.c_str());
		return -1;
	}

	struct stat st;
	fstat(model.mmap_fd, &st);
	model.mmap_size = st.st_size;
	model.mmap_addr = mmap(nullptr, model.mmap_size, PROT_READ, MAP_PRIVATE, model.mmap_fd, 0);
	if (model.mmap_addr == MAP_FAILED) {
		fprintf(stderr, "pflash: mmap failed for '%s'\n", gguf_path.c_str());
		close(model.mmap_fd);
		return -1;
	}

	// parse GGUF header
	struct gguf_init_params gparams = { /*.no_alloc =*/ true, /*.ctx =*/ &model.ctx_ggml };
	struct gguf_context * gctx = gguf_init_from_file(gguf_path.c_str(), gparams);
	if (!gctx) {
		fprintf(stderr, "pflash: failed to parse GGUF '%s'\n", gguf_path.c_str());
		pflash_model_free(model);
		return -1;
	}

	// read architecture metadata
	auto get_u32 = [&](const char * key, uint32_t def) -> uint32_t {
		int idx = gguf_find_key(gctx, key);
		return idx >= 0 ? gguf_get_val_u32(gctx, idx) : def;
	};
	auto get_f32 = [&](const char * key, float def) -> float {
		int idx = gguf_find_key(gctx, key);
		return idx >= 0 ? gguf_get_val_f32(gctx, idx) : def;
	};

	// Detect architecture: qwen3 (pure-attention) vs qwen35 (hybrid SSM+attention)
	int arch_key_idx = gguf_find_key(gctx, "general.architecture");
	const char * arch_str = arch_key_idx >= 0 ? gguf_get_val_str(gctx, arch_key_idx) : "qwen3";
	bool is_qwen35 = (strncmp(arch_str, "qwen35", 6) == 0);
	const char * pfx = is_qwen35 ? "qwen35" : "qwen3";

	// Helper: build "<arch>.<suffix>" key on the stack
	char pk_buf[128];
	auto pk = [&](const char * suffix) -> const char * {
		snprintf(pk_buf, sizeof(pk_buf), "%s.%s", pfx, suffix);
		return pk_buf;
	};

	model.n_embd     = (int)get_u32(pk("embedding_length"),       1024);
	model.n_heads    = (int)get_u32(pk("attention.head_count"),    16);
	model.n_kv_heads = (int)get_u32(pk("attention.head_count_kv"), 8);
	model.n_ff       = (int)get_u32(pk("feed_forward_length"),     3072);
	// key_length is the per-head QK dimension; may differ from n_embd/n_heads
	model.d_head          = (int)get_u32(pk("attention.key_length"), model.n_embd / model.n_heads);
	model.rope_freq_base  = get_f32(pk("rope.freq_base"), 1000000.0f);
	model.rope_type       = 2; // NEOX

	// vocab_size: prefer arch key; qwen35 GGUFs omit it — fall back to tokenizer array count
	{
		char vkey[128];
		snprintf(vkey, sizeof(vkey), "%s.vocab_size", pfx);
		int vidx = gguf_find_key(gctx, vkey);
		if (vidx >= 0) {
			model.n_vocab = (int)gguf_get_val_u32(gctx, vidx);
		} else {
			int tidx = gguf_find_key(gctx, "tokenizer.ggml.tokens");
			model.n_vocab = (tidx >= 0) ? (int)gguf_get_arr_n(gctx, tidx) : 151936;
		}
	}

	// full_attention_interval > 0: hybrid SSM+attention model; only every N-th layer
	// (0-indexed: layers where (i+1) % interval == 0) has separate Q/K/V attention weights.
	int full_attn_interval = is_qwen35 ? (int)get_u32("qwen35.full_attention_interval", 0) : 0;
	auto is_full_attn = [&](int i) -> bool {
		return full_attn_interval <= 0 || (i + 1) % full_attn_interval == 0;
	};

	// Count total physical layers, then count scoring (full-attention) layers
	int n_total_layers = (int)get_u32(pk("block_count"), 28);
	int n_scoring_layers = 0;
	for (int i = 0; i < n_total_layers; i++) {
		if (is_full_attn(i)) n_scoring_layers++;
	}
	model.n_layers = n_scoring_layers;

	fprintf(stderr, "pflash: %s scorer — %d scoring layers (/%d total), %d embd, %d heads (%d kv), d=%d, vocab=%d\n",
		arch_str, model.n_layers, n_total_layers, model.n_embd, model.n_heads, model.n_kv_heads, model.d_head, model.n_vocab);

	// If tok_embd is quantized, override its type to F32 before backend allocation so
	// ggml_get_rows works on all backends (quantized get_rows unsupported on GPU backends).
	// ggml_compute_forward_get_rows_f16 writes F32 output to an F16-sized buffer (crash),
	// so F32 is the correct target type. We dequantize in the tensor copy loop below.
	ggml_type tok_embd_orig_type = GGML_TYPE_F32;
	{
		ggml_tensor * te = ggml_get_tensor(model.ctx_ggml, "token_embd.weight");
		if (te && ggml_is_quantized(te->type)) {
			tok_embd_orig_type = te->type;
			te->type  = GGML_TYPE_F32;
			te->nb[0] = sizeof(float);
			te->nb[1] = te->ne[0] * te->nb[0];
			te->nb[2] = te->ne[1] * te->nb[1];
			te->nb[3] = te->ne[2] * te->nb[2];
			fprintf(stderr, "pflash: tok_embd %s → F32 dequant for get_rows compat\n",
					ggml_type_name(tok_embd_orig_type));
		}
	}

	// Phase 3: use ROCm GPU backend to store scorer weights in VRAM;
	// fall back to CPU if no GPU device is registered (e.g. Vulkan-only build).
	ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
	if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU_FULL); // Vulkan registers as GPU_FULL
	ggml_backend_t backend = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
	if (!backend) {
		fprintf(stderr, "pflash: no GPU device; falling back to CPU backend for scorer weights\n");
		backend = ggml_backend_cpu_init();
	}
	if (!backend) {
		fprintf(stderr, "pflash: cannot init backend for scorer\n");
		gguf_free(gctx);
		pflash_model_free(model);
		return -1;
	}
	(void)gpu_device;

	model.buf_gpu = ggml_backend_alloc_ctx_tensors(model.ctx_ggml, backend);
	if (!model.buf_gpu) {
		fprintf(stderr, "pflash: GPU allocation failed\n");
		ggml_backend_free(backend);
		gguf_free(gctx);
		pflash_model_free(model);
		return -1;
	}

	// copy tensor data from mmap to GPU
	const int n_tensors = gguf_get_n_tensors(gctx);
	for (int i = 0; i < n_tensors; i++) {
		const char * name = gguf_get_tensor_name(gctx, i);
		ggml_tensor * t = ggml_get_tensor(model.ctx_ggml, name);
		if (!t) continue;

		size_t offset = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, i);
		const void * src = (const char *)model.mmap_addr + offset;

		if (tok_embd_orig_type != GGML_TYPE_F32 && strcmp(name, "token_embd.weight") == 0) {
			// Bulk dequant: one CPU staging buffer → one GPU upload.
			// Replaces n_vocab separate ggml_backend_tensor_set calls (each with PCIe/HIP
			// fixed overhead) with a single large transfer.  For vocab=248320/n_embd=1024
			// this cuts tok_embd load from ~2.5s to <0.05s.
			int64_t n_embd  = t->ne[0];
			int64_t n_vocab = t->ne[1];
			size_t  qbytes  = ggml_row_size(tok_embd_orig_type, n_embd);
			const struct ggml_type_traits * traits = ggml_get_type_traits(tok_embd_orig_type);
			std::vector<float> f32buf((size_t)n_vocab * n_embd);
			for (int64_t r = 0; r < n_vocab; r++) {
				traits->to_float((const char *)src + r * qbytes, f32buf.data() + r * n_embd, n_embd);
			}
			ggml_backend_tensor_set(t, f32buf.data(), 0, (size_t)n_vocab * n_embd * sizeof(float));
		} else {
			ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
		}
	}

	// resolve tensor pointers
	model.tok_embd    = get_tensor(model.ctx_ggml, "token_embd.weight");
	model.output_norm = get_tensor(model.ctx_ggml, "output_norm.weight");
	model.output      = ggml_get_tensor(model.ctx_ggml, "output.weight");
	if (!model.output) {
		model.output = model.tok_embd; // tied embeddings
	}

	// qwen35 uses post_attention_norm for the FFN layer-norm; qwen3 uses ffn_norm
	const char * ffn_norm_fmt = is_qwen35 ? "blk.%d.post_attention_norm.weight"
	                                       : "blk.%d.ffn_norm.weight";

	model.layers.resize(model.n_layers);
	char buf[256];
	int fa_idx = 0;
	for (int i = 0; i < n_total_layers; i++) {
		if (!is_full_attn(i)) continue; // skip SSM-only layers in hybrid architectures
		auto & l = model.layers[fa_idx++];
		auto tn = [&](const char * fmt) -> ggml_tensor * {
			snprintf(buf, sizeof(buf), fmt, i);
			return get_tensor(model.ctx_ggml, buf);
		};
		l.attn_norm = tn("blk.%d.attn_norm.weight");
		l.wq        = tn("blk.%d.attn_q.weight");
		l.wk        = tn("blk.%d.attn_k.weight");
		l.wv        = tn("blk.%d.attn_v.weight");
		l.wo        = tn("blk.%d.attn_output.weight");
		l.q_norm    = tn("blk.%d.attn_q_norm.weight");
		l.k_norm    = tn("blk.%d.attn_k_norm.weight");
		l.ffn_norm  = tn(ffn_norm_fmt);
		l.ffn_gate  = tn("blk.%d.ffn_gate.weight");
		l.ffn_up    = tn("blk.%d.ffn_up.weight");
		l.ffn_down  = tn("blk.%d.ffn_down.weight");
	}

	ggml_backend_free(backend);
	gguf_free(gctx);

	return 0;
}

void pflash_model_free(pflash_model & model) {
	if (model.buf_gpu) {
		ggml_backend_buffer_free(model.buf_gpu);
		model.buf_gpu = nullptr;
	}
	if (model.ctx_ggml) {
		ggml_free(model.ctx_ggml);
		model.ctx_ggml = nullptr;
	}
	if (model.mmap_addr && model.mmap_addr != MAP_FAILED) {
		munmap(model.mmap_addr, model.mmap_size);
		model.mmap_addr = nullptr;
	}
	if (model.mmap_fd >= 0) {
		close(model.mmap_fd);
		model.mmap_fd = -1;
	}
	model.layers.clear();
}
