#include "pflash-loader.h"

#include "ggml.h"
#include "ggml-backend.h"
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

	model.n_embd     = (int)get_u32("qwen3.embedding_length", 1024);
	model.n_heads    = (int)get_u32("qwen3.attention.head_count", 16);
	model.n_kv_heads = (int)get_u32("qwen3.attention.head_count_kv", 8);
	model.n_layers   = (int)get_u32("qwen3.block_count", 28);
	model.n_ff       = (int)get_u32("qwen3.feed_forward_length", 3072);
	model.n_vocab    = (int)get_u32("qwen3.vocab_size", 151936);
	model.d_head     = model.n_embd / model.n_heads;
	model.rope_freq_base = get_f32("qwen3.rope.freq_base", 1000000.0f);
	model.rope_type  = 2; // NEOX

	fprintf(stderr, "pflash: Qwen3-0.6B loaded — %d layers, %d embd, %d heads (%d kv), %d ff, %d vocab\n",
		model.n_layers, model.n_embd, model.n_heads, model.n_kv_heads, model.n_ff, model.n_vocab);

	// allocate GPU buffer for all tensors
	// portable across ROCm/CUDA/Vulkan: probe by name in preference order
	const std::string gpu_dev_params = std::to_string(gpu_device);
	ggml_backend_dev_t bdev = ggml_backend_dev_by_name("CUDA");
	if (!bdev) bdev = ggml_backend_dev_by_name("ROCm");
	if (!bdev) bdev = ggml_backend_dev_by_name("Vulkan");
	ggml_backend_t backend = bdev ? ggml_backend_dev_init(bdev, gpu_dev_params.c_str()) : nullptr;
	if (!backend) {
		fprintf(stderr, "pflash: cannot init GPU backend (device %d)\n", gpu_device);
		gguf_free(gctx);
		pflash_model_free(model);
		return -1;
	}

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
		ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
	}

	// resolve tensor pointers
	model.tok_embd    = get_tensor(model.ctx_ggml, "token_embd.weight");
	model.output_norm = get_tensor(model.ctx_ggml, "output_norm.weight");
	model.output      = ggml_get_tensor(model.ctx_ggml, "output.weight");
	if (!model.output) {
		model.output = model.tok_embd; // tied embeddings
	}

	model.layers.resize(model.n_layers);
	char buf[256];
	for (int i = 0; i < model.n_layers; i++) {
		auto & l = model.layers[i];
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
		l.ffn_norm  = tn("blk.%d.ffn_norm.weight");
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
