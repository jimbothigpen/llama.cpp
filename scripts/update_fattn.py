import sys

def main():
    path = "ggml/src/ggml-cuda/fattn.cu"
    with open(path, "r") as f:
        content = f.read()
        
    old_code = """        const bool turbo_matched = (K->type == V->type &&
            (K->type == GGML_TYPE_TURBOQ4_0 || K->type == GGML_TYPE_TURBOQ3_0 || K->type == GGML_TYPE_TURBOQ2_0));
        if (ggml_cuda_turbo_mma_fused() && turbo_matched
                && Q->ne[1] <= 4 && V->ne[0] == Q->ne[0] && turing_mma_available(cc)) {
            if (Q->ne[0] == 128) {
                switch (K->type) {
                    case GGML_TYPE_TURBOQ4_0: ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<128, 128, GGML_TYPE_TURBOQ4_0, GGML_TYPE_TURBOQ4_0>(ctx, dst); return;
                    case GGML_TYPE_TURBOQ3_0: ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<128, 128, GGML_TYPE_TURBOQ3_0, GGML_TYPE_TURBOQ3_0>(ctx, dst); return;
                    case GGML_TYPE_TURBOQ2_0: ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<128, 128, GGML_TYPE_TURBOQ2_0, GGML_TYPE_TURBOQ2_0>(ctx, dst); return;
                    default: break;
                }
            }
            if (Q->ne[0] == 256) {
                switch (K->type) {
                    case GGML_TYPE_TURBOQ4_0: ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<256, 256, GGML_TYPE_TURBOQ4_0, GGML_TYPE_TURBOQ4_0>(ctx, dst); return;
                    case GGML_TYPE_TURBOQ3_0: ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<256, 256, GGML_TYPE_TURBOQ3_0, GGML_TYPE_TURBOQ3_0>(ctx, dst); return;
                    default: break;
                }
            }
        }"""
        
    new_code = """        const bool turbo_sym = (K->type == V->type &&
            (K->type == GGML_TYPE_TURBOQ4_0 || K->type == GGML_TYPE_TURBOQ3_0 || K->type == GGML_TYPE_TURBOQ2_0));
        const bool turbo_asym = (
            (K->type == GGML_TYPE_TURBOQ4_0 && (V->type == GGML_TYPE_TURBOQ3_0 || V->type == GGML_TYPE_TURBOQ2_0)) ||
            (K->type == GGML_TYPE_TURBOQ3_0 && V->type == GGML_TYPE_TURBOQ2_0)
        );
        const bool turbo_matched = turbo_sym || turbo_asym;
        if (ggml_cuda_turbo_mma_fused() && turbo_matched
                && Q->ne[1] <= 4 && V->ne[0] == Q->ne[0] && turing_mma_available(cc)) {
            if (Q->ne[0] == 128) {
                if (K->type == GGML_TYPE_TURBOQ4_0 && V->type == GGML_TYPE_TURBOQ4_0) { ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<128, 128, GGML_TYPE_TURBOQ4_0, GGML_TYPE_TURBOQ4_0>(ctx, dst); return; }
                if (K->type == GGML_TYPE_TURBOQ3_0 && V->type == GGML_TYPE_TURBOQ3_0) { ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<128, 128, GGML_TYPE_TURBOQ3_0, GGML_TYPE_TURBOQ3_0>(ctx, dst); return; }
                if (K->type == GGML_TYPE_TURBOQ2_0 && V->type == GGML_TYPE_TURBOQ2_0) { ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<128, 128, GGML_TYPE_TURBOQ2_0, GGML_TYPE_TURBOQ2_0>(ctx, dst); return; }
                if (K->type == GGML_TYPE_TURBOQ4_0 && V->type == GGML_TYPE_TURBOQ3_0) { ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<128, 128, GGML_TYPE_TURBOQ4_0, GGML_TYPE_TURBOQ3_0>(ctx, dst); return; }
                if (K->type == GGML_TYPE_TURBOQ4_0 && V->type == GGML_TYPE_TURBOQ2_0) { ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<128, 128, GGML_TYPE_TURBOQ4_0, GGML_TYPE_TURBOQ2_0>(ctx, dst); return; }
                if (K->type == GGML_TYPE_TURBOQ3_0 && V->type == GGML_TYPE_TURBOQ2_0) { ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<128, 128, GGML_TYPE_TURBOQ3_0, GGML_TYPE_TURBOQ2_0>(ctx, dst); return; }
            }
            if (Q->ne[0] == 256) {
                if (K->type == GGML_TYPE_TURBOQ4_0 && V->type == GGML_TYPE_TURBOQ4_0) { ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<256, 256, GGML_TYPE_TURBOQ4_0, GGML_TYPE_TURBOQ4_0>(ctx, dst); return; }
                if (K->type == GGML_TYPE_TURBOQ3_0 && V->type == GGML_TYPE_TURBOQ3_0) { ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<256, 256, GGML_TYPE_TURBOQ3_0, GGML_TYPE_TURBOQ3_0>(ctx, dst); return; }
                if (K->type == GGML_TYPE_TURBOQ2_0 && V->type == GGML_TYPE_TURBOQ2_0) { ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<256, 256, GGML_TYPE_TURBOQ2_0, GGML_TYPE_TURBOQ2_0>(ctx, dst); return; }
                if (K->type == GGML_TYPE_TURBOQ4_0 && V->type == GGML_TYPE_TURBOQ3_0) { ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<256, 256, GGML_TYPE_TURBOQ4_0, GGML_TYPE_TURBOQ3_0>(ctx, dst); return; }
                if (K->type == GGML_TYPE_TURBOQ4_0 && V->type == GGML_TYPE_TURBOQ2_0) { ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<256, 256, GGML_TYPE_TURBOQ4_0, GGML_TYPE_TURBOQ2_0>(ctx, dst); return; }
                if (K->type == GGML_TYPE_TURBOQ3_0 && V->type == GGML_TYPE_TURBOQ2_0) { ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<256, 256, GGML_TYPE_TURBOQ3_0, GGML_TYPE_TURBOQ2_0>(ctx, dst); return; }
            }
        }"""
        
    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(path, "w") as f:
            f.write(content)
        print("Successfully updated fattn.cu")
    else:
        print("Could not find the target code block in fattn.cu")

if __name__ == "__main__":
    main()
