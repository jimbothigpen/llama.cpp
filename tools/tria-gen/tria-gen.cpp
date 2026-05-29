/*
 * tria-gen.cpp — TRIA v1 calibration-stats generator
 *
 * Collects pre-RoPE Q activation statistics (per layer/head/frequency) from a
 * calibration corpus and writes a .tria v1 binary for use with the TriAttention
 * KV-cache eviction scorer (src/triattention.c).
 *
 * Targeting: Qwen3-8B-class models (28 layers, 32 Q heads / 8 KV heads,
 * head_dim=128, freq_count=64, rope_neox=true). Will work for any full-RoPE
 * Llama/Qwen architecture; fused-QKV (MLA) may lack a standalone Qcur tensor.
 *
 * Hookable tensor: "Qcur-{layer}" — the MUL_MAT result of the Q projection,
 * shape [n_embd_q, n_tokens] (2D, pre-reshape, pre-RoPE, pre-QKNorm for Qwen3).
 * The RESHAPE op is a view (skipped by the scheduler's eval callback), so we
 * capture the 2D MUL_MAT result directly. This is pre-RoPE: the mean Q direction
 * is position-invariant, unlike post-RoPE Q which averages to ~zero.
 *
 * Note (Qwen3 QKNorm): Qwen3 applies QKNorm AFTER the MUL_MAT and BEFORE RoPE.
 * We capture pre-QKNorm Q. The direction is similar but the magnitude is slightly
 * higher than post-norm. For v2+, collecting "Qcur_normed-{layer}" would be more
 * accurate; flagged for Option-1 pivot.
 *
 * Usage:
 *   llama-tria-gen -m model.gguf -f corpus.txt -o model.tria [-c 512]
 */

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>

/* Per-layer/head accumulation buffers (double precision for stability at 1M+ tokens) */
struct tria_acc {
    int32_t n_layers;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t head_dim;
    int32_t n_embd_q;     /* n_heads * head_dim */
    int32_t freq_count;
    int     rope_neox;
    bool    initialized;
    int64_t token_count;

    /* [n_layers * n_heads * freq_count], indexed (l * n_heads + h) * freq_count + f */
    std::vector<double> sum_real;
    std::vector<double> sum_imag;
    std::vector<double> sum_abs;
};

struct tria_cb_ctx {
    tria_acc         * acc;
    std::vector<float> buf;   /* scratch for GPU→CPU copy */
};

/* Parse layer index from tensor name "Qcur-N". Returns -1 if not a match. */
static int parse_qcur_layer(const char * name) {
    if (strncmp(name, "Qcur-", 5) != 0) return -1;
    char * end;
    long il = strtol(name + 5, &end, 10);
    if (end == name + 5 || *end != '\0' || il < 0) return -1;
    return (int)il;
}

static bool tria_collect_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    tria_cb_ctx * cbctx = (tria_cb_ctx *)user_data;
    tria_acc    * acc   = cbctx->acc;

    if (!acc->initialized) return false;

    int il = parse_qcur_layer(t->name);
    if (il < 0 || il >= acc->n_layers) return false;

    /*
     * Select the 2D MUL_MAT result (pre-RoPE):
     *   ne[0] == n_embd_q  (n_heads * head_dim)
     *   ne[1] == n_tokens  (batch size, > 0)
     *   ne[2] == 1         (2D tensor)
     *
     * The RESHAPE (3D) op is skipped by the scheduler's eval callback.
     * The post-RoPE ROPE op has shape [head_dim, n_heads, n_tokens],
     * so ne[0] == head_dim != n_embd_q. This filter is unambiguous.
     */
    if (t->ne[0] != acc->n_embd_q || t->ne[2] != 1) return false;
    int64_t n_tokens = t->ne[1];
    if (n_tokens <= 0) return false;

    if (ask) return true;

    /* ask == false: data is ready; copy to host if needed */
    int     fc      = acc->freq_count;
    int     nh      = acc->n_heads;
    int32_t n_embd  = acc->n_embd_q;   /* = nh * head_dim */
    int32_t hd      = acc->head_dim;
    size_t  nbytes  = (size_t)n_tokens * n_embd * sizeof(float);

    cbctx->buf.resize((size_t)n_tokens * n_embd);
    ggml_backend_tensor_get(t, cbctx->buf.data(), 0, nbytes);

    /*
     * 2D layout: element [dim, tok] at buf[tok * n_embd + dim]
     * Head h occupies dims [h*hd .. (h+1)*hd).
     * NEOX split: real[f] = dim h*hd+f, imag[f] = dim h*hd+fc+f
     * NORM split: real[f] = dim h*hd+2f, imag[f] = dim h*hd+2f+1
     */
    for (int64_t tok = 0; tok < n_tokens; tok++) {
        const float * row = cbctx->buf.data() + tok * n_embd;
        for (int h = 0; h < nh; h++) {
            const float * q = row + h * hd;
            int base = (il * nh + h) * fc;
            if (acc->rope_neox) {
                for (int f = 0; f < fc; f++) {
                    float r  = q[f];
                    float im = q[fc + f];
                    acc->sum_real[base + f] += (double)r;
                    acc->sum_imag[base + f] += (double)im;
                    acc->sum_abs [base + f] += (double)sqrtf(r*r + im*im);
                }
            } else {
                for (int f = 0; f < fc; f++) {
                    float r  = q[2*f];
                    float im = q[2*f + 1];
                    acc->sum_real[base + f] += (double)r;
                    acc->sum_imag[base + f] += (double)im;
                    acc->sum_abs [base + f] += (double)sqrtf(r*r + im*im);
                }
            }
        }
    }

    /* Count tokens from layer 0 only (each batch fires once per layer) */
    if (il == 0) {
        acc->token_count += n_tokens;
    }

    return true;
}

static void print_usage(int, char ** argv) {
    LOG("\nUsage:\n  %s -m model.gguf -f corpus.txt -o out.tria [-c 512]\n\n", argv[0]);
    LOG("  -m MODEL   Path to model GGUF file\n");
    LOG("  -f FILE    Calibration corpus text file\n");
    LOG("  -o FILE    Output .tria path (default: output.tria)\n");
    LOG("  -c N       Chunk size in tokens (default: 512)\n");
    LOG("\n");
}

static bool write_tria_v1(
    const char * out_path,
    tria_acc   * acc,
    float        rope_theta,
    float        attn_scale
) {
    int64_t tc = acc->token_count;
    if (tc == 0) {
        fprintf(stderr, "tria-gen: no tokens collected\n");
        return false;
    }

    FILE * fp = fopen(out_path, "wb");
    if (!fp) { perror(out_path); return false; }

    int nl  = acc->n_layers;
    int nh  = acc->n_heads;
    int nkv = acc->n_kv_heads;
    int hd  = acc->head_dim;
    int fc  = acc->freq_count;

    /*
     * 64-byte header layout (triattention.c:22-159):
     * [0]   magic         4   = 0x54524941
     * [4]   version       4   = 1
     * [8]   num_layers    4
     * [12]  num_heads     4
     * [16]  num_kv_heads  4
     * [20]  head_dim      4
     * [24]  freq_count    4
     * [28]  rope_theta    4 (float32)
     * [32]  attn_scale    4 (float32)
     * [36]  nonrot_dim    4   = 0 (v1; loader reads only for version>=3)
     * [40-63] reserved   24
     * Total: 64 bytes; loader seeks to TRIA_HEADER_SIZE (64) before body.
     */
    uint32_t magic    = 0x54524941u;
    uint32_t version  = 1u;
    uint32_t unl      = (uint32_t)nl;
    uint32_t unh      = (uint32_t)nh;
    uint32_t unkv     = (uint32_t)nkv;
    uint32_t uhd      = (uint32_t)hd;
    uint32_t ufc      = (uint32_t)fc;
    uint32_t nonrot   = 0u;

    fwrite(&magic,     4, 1, fp);
    fwrite(&version,   4, 1, fp);
    fwrite(&unl,       4, 1, fp);
    fwrite(&unh,       4, 1, fp);
    fwrite(&unkv,      4, 1, fp);
    fwrite(&uhd,       4, 1, fp);
    fwrite(&ufc,       4, 1, fp);
    fwrite(&rope_theta, 4, 1, fp);
    fwrite(&attn_scale, 4, 1, fp);
    fwrite(&nonrot,    4, 1, fp);   /* nonrot_dim = 0 */

    uint8_t pad[24] = {};
    fwrite(pad, 1, sizeof(pad), fp);   /* pad to offset 64 */

    /* v1: no layer_budget_scales (loader fills 1.0f for version < 2) */

    /*
     * Per-head body (layer-major, head-minor):
     *   q_mean_real [fc] float32
     *   q_mean_imag [fc] float32
     *   q_abs_mean  [fc] float32
     *   mrl         [fc] float32 = 0 (field unused by scorer; loader skips via fseek)
     * Each entry: 4 * fc * 4 = 1024 bytes (for fc=64)
     * Total body: nl*nh * 1024 = 28*32*1024 = 917,504 bytes
     * File total: 64 + 917,504 = 917,568 bytes
     */
    std::vector<float> mrl(fc, 0.0f);
    std::vector<float> tmp(fc);

    for (int l = 0; l < nl; l++) {
        for (int h = 0; h < nh; h++) {
            int base = (l * nh + h) * fc;

            for (int f = 0; f < fc; f++)
                tmp[f] = (float)(acc->sum_real[base + f] / (double)tc);
            fwrite(tmp.data(), 4, fc, fp);   /* q_mean_real */

            for (int f = 0; f < fc; f++)
                tmp[f] = (float)(acc->sum_imag[base + f] / (double)tc);
            fwrite(tmp.data(), 4, fc, fp);   /* q_mean_imag */

            for (int f = 0; f < fc; f++)
                tmp[f] = (float)(acc->sum_abs[base + f] / (double)tc);
            fwrite(tmp.data(), 4, fc, fp);   /* q_abs_mean */

            fwrite(mrl.data(), 4, fc, fp);   /* mrl (zeros, skipped by loader) */
        }
    }

    long final_size = ftell(fp);
    fclose(fp);

    fprintf(stderr,
        "tria-gen: wrote %ld bytes → %s "
        "(v1, %d layers, %d heads/%d kv, fc=%d, %" PRId64 " tokens)\n",
        final_size, out_path, nl, nh, nkv, fc, tc);
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx    = 512;
    params.escape   = false;
    params.warmup   = false;
    params.out_file = "output.tria";

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }

    if (params.prompt.empty()) {
        fprintf(stderr, "tria-gen: no corpus provided — use -f corpus.txt\n");
        print_usage(argc, argv);
        return 1;
    }

    const int n_ctx = (params.n_ctx > 0) ? params.n_ctx : 512;

    llama_backend_init();
    llama_numa_init(params.numa);

    /*
     * Set up callback context before common_init_from_params.
     * The eval callback fires only during llama_decode (not during init),
     * so acc.initialized=false safely prevents any premature accumulation.
     */
    tria_acc     acc    = {};
    tria_cb_ctx  cbctx  = { &acc, {} };
    params.cb_eval           = tria_collect_cb;
    params.cb_eval_user_data = &cbctx;

    auto llama_init = common_init_from_params(params);
    auto * model    = llama_init->model();
    auto * ctx      = llama_init->context();

    if (!model || !ctx) {
        fprintf(stderr, "tria-gen: failed to load model/context\n");
        return 1;
    }

    /* Read model hparams via public API */
    acc.n_layers   = llama_model_n_layer(model);
    acc.n_heads    = llama_model_n_head(model);
    acc.n_kv_heads = llama_model_n_head_kv(model);
    acc.head_dim   = llama_model_n_embd(model) / acc.n_heads;
    acc.n_embd_q   = acc.n_heads * acc.head_dim;
    acc.freq_count = acc.head_dim / 2;   /* full-RoPE assumption: freq_count = head_dim/2 */

    /* rope_neox: NEOX/IMROPE = split-half; NORM = interleaved */
    {
        enum llama_rope_type rt = llama_model_rope_type(model);
        acc.rope_neox = (rt != LLAMA_ROPE_TYPE_NORM) ? 1 : 0;
    }

    acc.initialized  = false;   /* will enable after buffers allocated */
    acc.token_count  = 0;
    size_t buf_elems = (size_t)acc.n_layers * acc.n_heads * acc.freq_count;
    acc.sum_real.assign(buf_elems, 0.0);
    acc.sum_imag.assign(buf_elems, 0.0);
    acc.sum_abs .assign(buf_elems, 0.0);
    acc.initialized  = true;

    /* Read rope_theta from model metadata: key = "<arch>.rope.freq_base" */
    float rope_theta = 1000000.0f;   /* Qwen3 default; overwritten if found in metadata */
    {
        char arch[128] = {};
        if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) >= 0) {
            char key[256];
            snprintf(key, sizeof(key), "%s.rope.freq_base", arch);
            char val[64] = {};
            if (llama_model_meta_val_str(model, key, val, sizeof(val)) >= 0) {
                float v = strtof(val, nullptr);
                if (v > 0.0f && isfinite(v)) rope_theta = v;
            }
        }
    }
    float attn_scale = 1.0f / sqrtf((float)acc.head_dim);

    fprintf(stderr,
        "tria-gen: layers=%d heads=%d kv=%d head_dim=%d fc=%d\n",
        acc.n_layers, acc.n_heads, acc.n_kv_heads, acc.head_dim, acc.freq_count);
    fprintf(stderr,
        "tria-gen: rope_neox=%d rope_theta=%.0f attn_scale=%.6f\n",
        acc.rope_neox, rope_theta, attn_scale);

    /* Validate dimensions satisfy the consumer constraint: 2*freq_count <= head_dim */
    if (2 * acc.freq_count > acc.head_dim) {
        fprintf(stderr,
            "tria-gen: ERROR freq_count=%d exceeds head_dim/2=%d — partial-RoPE model?\n",
            acc.freq_count, acc.head_dim / 2);
        fprintf(stderr,
            "tria-gen: partial-RoPE models require v3 generator (Option 1). Aborting.\n");
        return 1;
    }

    /* Tokenize corpus */
    std::vector<llama_token> tokens =
        common_tokenize(ctx, params.prompt, /* add_special */ true, /* parse_special */ false);

    fprintf(stderr, "tria-gen: corpus = %zu tokens; chunk_size = %d\n", tokens.size(), n_ctx);

    if ((int)tokens.size() < n_ctx) {
        fprintf(stderr, "tria-gen: corpus too short (need >= %d tokens)\n", n_ctx);
        return 1;
    }

    const int n_chunks = (int)(tokens.size() / n_ctx);
    fprintf(stderr, "tria-gen: processing %d chunks\n", n_chunks);

    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

    for (int ci = 0; ci < n_chunks; ci++) {
        llama_memory_clear(llama_get_memory(ctx), true);

        const int start  = ci * n_ctx;
        batch.n_tokens   = n_ctx;

        for (int i = 0; i < n_ctx; i++) {
            batch.token[i]     = tokens[start + i];
            batch.pos[i]       = (llama_pos)i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i == n_ctx - 1);   /* only last token needs logits */
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "tria-gen: llama_decode failed at chunk %d\n", ci);
            llama_batch_free(batch);
            return 1;
        }

        if ((ci + 1) % 20 == 0 || ci == n_chunks - 1) {
            fprintf(stderr,
                "tria-gen: chunk %d/%d  (%" PRId64 " tokens accumulated)\n",
                ci + 1, n_chunks, acc.token_count);
        }
    }

    llama_batch_free(batch);

    if (!write_tria_v1(params.out_file.c_str(), &acc, rope_theta, attn_scale)) {
        return 1;
    }

    llama_backend_free();
    return 0;
}
