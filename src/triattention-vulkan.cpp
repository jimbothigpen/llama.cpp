/*
 * triattention-vulkan.cpp — Vulkan compute-shader port of the TriAttention
 * Phase-C GPU GQA scoring kernel (template: triattention-hip.hip).
 *
 * This is a fully self-contained Vulkan compute context, deliberately mirroring
 * the isolation of the `tria-hip` static library: it creates its own
 * VkInstance/VkDevice/queue/pipelines and manages its own VkBuffers, so it does
 * not depend on ggml-vulkan internals and works regardless of GGML_BACKEND_DL.
 * This is sound because the per-layer K is captured into CPU-backend buffers
 * (llama-triattention.cpp), so all K/stats data crosses the host boundary — no
 * sharing of ggml GPU allocations is required.
 *
 * Opaque "device pointers": the ABI hands persistent device buffers back to the
 * runtime as float* (omega/q_mean/global_scores). We allocate a small heap
 * VkBufHandle and reinterpret its address as float*; the runtime only null-checks
 * and round-trips these, never dereferencing them as floats.
 */

#include "triattention-vulkan.h"

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <vector>

/* Embedded SPIR-V (generated at build time by glslc -mfmt=num). */
static const uint32_t spv_raw_score[] = {
#include "tria_raw_score.comp.spv.inc"
};
static const uint32_t spv_znorm_agg[] = {
#include "tria_znorm_agg.comp.spv.inc"
};
static const uint32_t spv_znorm_global[] = {
#include "tria_znorm_global.comp.spv.inc"
};

#define QK8       32
#define BLK_BYTES 34   /* fp16 scale (2) + 32 int8 */
#define REDUCE_WG 256

/* ── opaque device buffer handle ─────────────────────────────────────── */
struct VkBufHandle {
    VkBuffer       buf  = VK_NULL_HANDLE;
    VkDeviceMemory mem  = VK_NULL_HANDLE;
    VkDeviceSize   bytes = 0;
};

/* ── push-constant blocks (must match the .comp layouts) ─────────────── */
struct RawPush {
    int n_tokens, cur_pos, n_embd_k_gqa, kvi, head_dim;
    int freq_count, rope_neox, n_offsets, q_off, use_qab;
};
struct AggPush    { int n_tokens; int is_first; };
struct GlobalPush { int n_tokens; float layer_weight; };

/* ── global Vulkan context ───────────────────────────────────────────── */
struct TriaVk {
    int                state = 0;            /* 0=uninit, 1=ready, -1=failed */
    VkInstance         instance = VK_NULL_HANDLE;
    VkPhysicalDevice   phys = VK_NULL_HANDLE;
    VkDevice           dev = VK_NULL_HANDLE;
    uint32_t           qfam = 0;
    VkQueue            queue = VK_NULL_HANDLE;
    VkCommandPool      cmdpool = VK_NULL_HANDLE;
    VkCommandBuffer    cmd = VK_NULL_HANDLE;
    VkFence            fence = VK_NULL_HANDLE;
    VkDescriptorPool   descpool = VK_NULL_HANDLE;
    uint32_t           host_mem_type = UINT32_MAX;

    VkDescriptorSetLayout dsl_raw = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_agg = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_global = VK_NULL_HANDLE;
    VkPipelineLayout      pl_raw = VK_NULL_HANDLE;
    VkPipelineLayout      pl_agg = VK_NULL_HANDLE;
    VkPipelineLayout      pl_global = VK_NULL_HANDLE;
    VkPipeline            p_raw = VK_NULL_HANDLE;
    VkPipeline            p_agg = VK_NULL_HANDLE;
    VkPipeline            p_global = VK_NULL_HANDLE;
};

static TriaVk    g_vk;
static std::mutex g_vk_mutex;

/* ── helpers ─────────────────────────────────────────────────────────── */

static VkDescriptorSetLayout make_dsl(VkDevice dev, uint32_t n_bindings) {
    std::vector<VkDescriptorSetLayoutBinding> b(n_bindings);
    for (uint32_t i = 0; i < n_bindings; i++) {
        b[i] = {};
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = n_bindings;
    ci.pBindings = b.data();
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(dev, &ci, nullptr, &dsl) != VK_SUCCESS) return VK_NULL_HANDLE;
    return dsl;
}

static bool make_pipeline(VkDevice dev, const uint32_t * spv, size_t spv_bytes,
                          VkDescriptorSetLayout dsl, uint32_t push_size,
                          VkPipelineLayout * out_pl, VkPipeline * out_p) {
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spv_bytes;
    smci.pCode = spv;
    VkShaderModule sm = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev, &smci, nullptr, &sm) != VK_SUCCESS) return false;

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = push_size;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(dev, &plci, nullptr, out_pl) != VK_SUCCESS) {
        vkDestroyShaderModule(dev, sm, nullptr);
        return false;
    }

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = *out_pl;
    const VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, out_p);
    vkDestroyShaderModule(dev, sm, nullptr);
    return r == VK_SUCCESS;
}

/* Lazily initialize the Vulkan device + pipelines. Returns true if ready. */
static bool vk_ensure_init() {
    if (g_vk.state == 1) return true;
    if (g_vk.state == -1) return false;
    g_vk.state = -1;  /* assume failure until we reach the end */

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "tria-vulkan";
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &g_vk.instance) != VK_SUCCESS) {
        fprintf(stderr, "tria-vk: vkCreateInstance failed\n");
        return false;
    }

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(g_vk.instance, &ndev, nullptr);
    if (ndev == 0) { fprintf(stderr, "tria-vk: no Vulkan devices\n"); return false; }
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(g_vk.instance, &ndev, devs.data());

    /* Prefer a real GPU (discrete, then integrated), skip CPU/llvmpipe. */
    VkPhysicalDevice pick = VK_NULL_HANDLE;
    int best = -1;
    for (auto d : devs) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        int score;
        switch (p.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   score = 3; break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score = 2; break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    score = 1; break;
            default:                                     score = 0; break;  /* CPU/other */
        }
        if (score > best) { best = score; pick = d; }
    }
    if (pick == VK_NULL_HANDLE) return false;
    g_vk.phys = pick;
    {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pick, &p);
        fprintf(stderr, "tria-vk: using device '%s'\n", p.deviceName);
    }

    /* Compute queue family. */
    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pick, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qfp(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(pick, &nqf, qfp.data());
    bool found_q = false;
    for (uint32_t i = 0; i < nqf; i++) {
        if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_vk.qfam = i; found_q = true; break; }
    }
    if (!found_q) { fprintf(stderr, "tria-vk: no compute queue\n"); return false; }

    /* Logical device with fp64 (needed for accurate trig range reduction). */
    VkPhysicalDeviceFeatures feats{};
    feats.shaderFloat64 = VK_TRUE;
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = g_vk.qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &feats;
    if (vkCreateDevice(pick, &dci, nullptr, &g_vk.dev) != VK_SUCCESS) {
        fprintf(stderr, "tria-vk: vkCreateDevice failed\n");
        return false;
    }
    vkGetDeviceQueue(g_vk.dev, g_vk.qfam, 0, &g_vk.queue);

    /* Host-visible + coherent memory type (APU targets: also device-local). */
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pick, &mp);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t best_mt = UINT32_MAX; int best_mt_score = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & want) != want) continue;
        const int sc = (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 1 : 0;
        if (sc > best_mt_score) { best_mt_score = sc; best_mt = i; }
    }
    if (best_mt == UINT32_MAX) { fprintf(stderr, "tria-vk: no host-visible memory\n"); return false; }
    g_vk.host_mem_type = best_mt;

    /* Command pool + buffer + fence. */
    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = g_vk.qfam;
    if (vkCreateCommandPool(g_vk.dev, &cpci, nullptr, &g_vk.cmdpool) != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = g_vk.cmdpool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_vk.dev, &cbai, &g_vk.cmd) != VK_SUCCESS) return false;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(g_vk.dev, &fci, nullptr, &g_vk.fence) != VK_SUCCESS) return false;

    /* Descriptor pool (reset per scoring call; 3 sets, <=12 storage buffers). */
    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 32;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 8;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(g_vk.dev, &dpci, nullptr, &g_vk.descpool) != VK_SUCCESS) return false;

    /* Descriptor set layouts + pipelines. */
    g_vk.dsl_raw    = make_dsl(g_vk.dev, 8);
    g_vk.dsl_agg    = make_dsl(g_vk.dev, 2);
    g_vk.dsl_global = make_dsl(g_vk.dev, 2);
    if (!g_vk.dsl_raw || !g_vk.dsl_agg || !g_vk.dsl_global) return false;

    if (!make_pipeline(g_vk.dev, spv_raw_score, sizeof(spv_raw_score),
                       g_vk.dsl_raw, sizeof(RawPush), &g_vk.pl_raw, &g_vk.p_raw)) {
        fprintf(stderr, "tria-vk: raw_score pipeline failed\n"); return false;
    }
    if (!make_pipeline(g_vk.dev, spv_znorm_agg, sizeof(spv_znorm_agg),
                       g_vk.dsl_agg, sizeof(AggPush), &g_vk.pl_agg, &g_vk.p_agg)) {
        fprintf(stderr, "tria-vk: znorm_agg pipeline failed\n"); return false;
    }
    if (!make_pipeline(g_vk.dev, spv_znorm_global, sizeof(spv_znorm_global),
                       g_vk.dsl_global, sizeof(GlobalPush), &g_vk.pl_global, &g_vk.p_global)) {
        fprintf(stderr, "tria-vk: znorm_global pipeline failed\n"); return false;
    }

    g_vk.state = 1;
    return true;
}

static bool vk_make_buffer(VkDeviceSize bytes, VkBufHandle * out) {
    out->buf = VK_NULL_HANDLE; out->mem = VK_NULL_HANDLE; out->bytes = bytes;
    if (bytes == 0) bytes = 4;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(g_vk.dev, &bci, nullptr, &out->buf) != VK_SUCCESS) return false;

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_vk.dev, out->buf, &mr);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = g_vk.host_mem_type;
    if (vkAllocateMemory(g_vk.dev, &mai, nullptr, &out->mem) != VK_SUCCESS) {
        vkDestroyBuffer(g_vk.dev, out->buf, nullptr); out->buf = VK_NULL_HANDLE; return false;
    }
    if (vkBindBufferMemory(g_vk.dev, out->buf, out->mem, 0) != VK_SUCCESS) {
        vkFreeMemory(g_vk.dev, out->mem, nullptr); vkDestroyBuffer(g_vk.dev, out->buf, nullptr);
        out->buf = VK_NULL_HANDLE; out->mem = VK_NULL_HANDLE; return false;
    }
    return true;
}

static void vk_free_buffer(VkBufHandle * h) {
    if (!h) return;
    if (h->mem) vkFreeMemory(g_vk.dev, h->mem, nullptr);
    if (h->buf) vkDestroyBuffer(g_vk.dev, h->buf, nullptr);
    h->buf = VK_NULL_HANDLE; h->mem = VK_NULL_HANDLE;
}

/* Upload host -> buffer (write n bytes from src at offset 0). */
static bool vk_upload(VkBufHandle * h, const void * src, size_t n) {
    if (n == 0) return true;
    void * p = nullptr;
    if (vkMapMemory(g_vk.dev, h->mem, 0, n, 0, &p) != VK_SUCCESS) return false;
    memcpy(p, src, n);
    vkUnmapMemory(g_vk.dev, h->mem);
    return true;
}
static bool vk_download(void * dst, VkBufHandle * h, size_t n) {
    if (n == 0) return true;
    void * p = nullptr;
    if (vkMapMemory(g_vk.dev, h->mem, 0, n, 0, &p) != VK_SUCCESS) return false;
    memcpy(dst, p, n);
    vkUnmapMemory(g_vk.dev, h->mem);
    return true;
}

static void bind_set(VkDescriptorSet set, const std::vector<VkBuffer> & bufs) {
    std::vector<VkDescriptorBufferInfo> infos(bufs.size());
    std::vector<VkWriteDescriptorSet>   writes(bufs.size());
    for (size_t i = 0; i < bufs.size(); i++) {
        infos[i] = {}; infos[i].buffer = bufs[i]; infos[i].offset = 0; infos[i].range = VK_WHOLE_SIZE;
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(g_vk.dev, (uint32_t)writes.size(), writes.data(), 0, nullptr);
}

static void full_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}

/* ── ABI: stats upload / free ────────────────────────────────────────── */

extern "C" int tria_vk_stats_upload(
        const float * omega, int freq_count,
        const float * q_mean_real, const float * q_mean_imag,
        int n_kv_heads,
        float ** omega_dev_out,
        float ** q_mean_real_dev_out,
        float ** q_mean_imag_dev_out) {
    std::lock_guard<std::mutex> lk(g_vk_mutex);
    if (!vk_ensure_init()) return -1;

    const size_t omega_bytes = (size_t)freq_count * sizeof(float);
    const size_t qm_bytes    = (size_t)n_kv_heads * freq_count * sizeof(float);

    VkBufHandle * h_omega = nullptr;
    VkBufHandle * h_qmr   = nullptr;

    if (omega_dev_out && omega && freq_count > 0) {
        h_omega = new VkBufHandle();
        if (!vk_make_buffer(omega_bytes, h_omega) || !vk_upload(h_omega, omega, omega_bytes)) {
            vk_free_buffer(h_omega); delete h_omega; return -1;
        }
        *omega_dev_out = reinterpret_cast<float*>(h_omega);
    }
    if (q_mean_real_dev_out && q_mean_real && qm_bytes > 0) {
        h_qmr = new VkBufHandle();
        if (!vk_make_buffer(qm_bytes, h_qmr) || !vk_upload(h_qmr, q_mean_real, qm_bytes)) {
            vk_free_buffer(h_qmr); delete h_qmr;
            if (h_omega) { vk_free_buffer(h_omega); delete h_omega; *omega_dev_out = nullptr; }
            return -1;
        }
        *q_mean_real_dev_out = reinterpret_cast<float*>(h_qmr);
    }
    if (q_mean_imag_dev_out && q_mean_imag && qm_bytes > 0) {
        VkBufHandle * h_qmi = new VkBufHandle();
        if (!vk_make_buffer(qm_bytes, h_qmi) || !vk_upload(h_qmi, q_mean_imag, qm_bytes)) {
            vk_free_buffer(h_qmi); delete h_qmi;
            if (h_omega) { vk_free_buffer(h_omega); delete h_omega; *omega_dev_out = nullptr; }
            if (h_qmr)   { vk_free_buffer(h_qmr);   delete h_qmr;   *q_mean_real_dev_out = nullptr; }
            return -1;
        }
        *q_mean_imag_dev_out = reinterpret_cast<float*>(h_qmi);
    }
    return 0;
}

extern "C" void tria_vk_stats_free(float * a, float * b, float * c) {
    std::lock_guard<std::mutex> lk(g_vk_mutex);
    if (g_vk.state != 1) return;
    float * arr[3] = { a, b, c };
    for (int i = 0; i < 3; i++) {
        if (!arr[i]) continue;
        VkBufHandle * h = reinterpret_cast<VkBufHandle*>(arr[i]);
        vk_free_buffer(h);
        delete h;
    }
}

extern "C" int tria_vk_scores_download(float * scores, const float * scores_dev, int n_scores) {
    if (!scores || !scores_dev || n_scores <= 0) return 0;
    std::lock_guard<std::mutex> lk(g_vk_mutex);
    if (g_vk.state != 1) return -1;
    VkBufHandle * h = reinterpret_cast<VkBufHandle*>(const_cast<float*>(scores_dev));
    return vk_download(scores, h, (size_t)n_scores * sizeof(float)) ? 0 : -1;
}

/* ── ABI: GPU scoring ────────────────────────────────────────────────── */

extern "C" int tria_vk_score_q8_0(
        const void * k_data_host,
        int n_tokens,
        int score_start,
        int cur_pos,
        int n_embd_k_gqa,
        int n_kv_heads,
        int n_heads,
        int head_dim,
        int freq_count,
        int rope_neox,
        const int * key_pos,
        const float * omega_dev,
        const float * q_mean_real_dev,
        const float * q_mean_imag_dev,
        const float * q_abs_mean_dev,
        int q_mean_offset,
        float layer_weight,
        float * global_scores_dev,
        int n_offsets,
        const int * offsets) {

    if (!k_data_host || n_tokens <= 0 || n_kv_heads <= 0) return 0;
    if (n_heads < n_kv_heads || (n_heads % n_kv_heads) != 0) return -1;
    if (freq_count > 64 || head_dim > 128 || (head_dim % QK8) != 0) return -1;
    if (!omega_dev || !q_mean_real_dev || !q_mean_imag_dev || !global_scores_dev) return -1;

    std::lock_guard<std::mutex> lk(g_vk_mutex);
    if (g_vk.state != 1) return -1;

    const int gqa = n_heads / n_kv_heads;
    const int row_bytes = (n_embd_k_gqa / QK8) * BLK_BYTES;
    size_t k_slice_bytes = (size_t)n_tokens * row_bytes;
    size_t k_buf_bytes   = (k_slice_bytes + 3) & ~(size_t)3;   /* round up for uint SSBO */

    VkBufHandle d_k{}, d_key{}, d_off{}, d_raw{}, d_agg{};
    int rc = -1;
    if (!vk_make_buffer(k_buf_bytes, &d_k))                              goto cleanup;
    if (!vk_make_buffer((size_t)n_tokens * sizeof(int),   &d_key))       goto cleanup;
    if (!vk_make_buffer((size_t)n_offsets * sizeof(int),  &d_off))       goto cleanup;
    if (!vk_make_buffer((size_t)n_tokens * sizeof(float), &d_raw))       goto cleanup;
    if (!vk_make_buffer((size_t)n_tokens * sizeof(float), &d_agg))       goto cleanup;

    if (!vk_upload(&d_k, (const uint8_t*)k_data_host + (size_t)score_start * row_bytes, k_slice_bytes)) goto cleanup;
    if (!vk_upload(&d_key, key_pos, (size_t)n_tokens * sizeof(int)))     goto cleanup;
    if (!vk_upload(&d_off, offsets, (size_t)n_offsets * sizeof(int)))    goto cleanup;

    {
        VkBufHandle * h_omega  = reinterpret_cast<VkBufHandle*>(const_cast<float*>(omega_dev));
        VkBufHandle * h_qmr    = reinterpret_cast<VkBufHandle*>(const_cast<float*>(q_mean_real_dev));
        VkBufHandle * h_qmi    = reinterpret_cast<VkBufHandle*>(const_cast<float*>(q_mean_imag_dev));
        VkBufHandle * h_qab    = q_abs_mean_dev
            ? reinterpret_cast<VkBufHandle*>(const_cast<float*>(q_abs_mean_dev)) : h_qmr; /* placeholder */
        VkBufHandle * h_global = reinterpret_cast<VkBufHandle*>(global_scores_dev);
        const int use_qab = q_abs_mean_dev ? 1 : 0;

        /* allocate 3 descriptor sets */
        vkResetDescriptorPool(g_vk.dev, g_vk.descpool, 0);
        VkDescriptorSetLayout layouts[3] = { g_vk.dsl_raw, g_vk.dsl_agg, g_vk.dsl_global };
        VkDescriptorSet sets[3] = {};
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = g_vk.descpool;
        dsai.descriptorSetCount = 3;
        dsai.pSetLayouts = layouts;
        if (vkAllocateDescriptorSets(g_vk.dev, &dsai, sets) != VK_SUCCESS) goto cleanup;

        bind_set(sets[0], { d_k.buf, d_key.buf, h_omega->buf, h_qmr->buf, h_qmi->buf,
                            h_qab->buf, d_off.buf, d_raw.buf });
        bind_set(sets[1], { d_raw.buf, d_agg.buf });
        bind_set(sets[2], { d_agg.buf, h_global->buf });

        /* record */
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandBuffer(g_vk.cmd, 0);
        vkBeginCommandBuffer(g_vk.cmd, &bi);

        const uint32_t raw_groups = (uint32_t)((n_tokens + 63) / 64);

        for (int kvi = 0; kvi < n_kv_heads; kvi++) {
            for (int g = 0; g < gqa; g++) {
                const int qh = kvi * gqa + g;
                RawPush rp{};
                rp.n_tokens = n_tokens; rp.cur_pos = cur_pos; rp.n_embd_k_gqa = n_embd_k_gqa;
                rp.kvi = kvi; rp.head_dim = head_dim; rp.freq_count = freq_count;
                rp.rope_neox = rope_neox; rp.n_offsets = n_offsets;
                rp.q_off = q_mean_offset + qh * freq_count; rp.use_qab = use_qab;
                vkCmdBindPipeline(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.p_raw);
                vkCmdBindDescriptorSets(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pl_raw,
                                        0, 1, &sets[0], 0, nullptr);
                vkCmdPushConstants(g_vk.cmd, g_vk.pl_raw, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(rp), &rp);
                vkCmdDispatch(g_vk.cmd, raw_groups, 1, 1);
                full_barrier(g_vk.cmd);

                AggPush ap{}; ap.n_tokens = n_tokens; ap.is_first = (g == 0) ? 1 : 0;
                vkCmdBindPipeline(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.p_agg);
                vkCmdBindDescriptorSets(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pl_agg,
                                        0, 1, &sets[1], 0, nullptr);
                vkCmdPushConstants(g_vk.cmd, g_vk.pl_agg, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(ap), &ap);
                vkCmdDispatch(g_vk.cmd, 1, 1, 1);
                full_barrier(g_vk.cmd);
            }
            GlobalPush gp{}; gp.n_tokens = n_tokens; gp.layer_weight = layer_weight;
            vkCmdBindPipeline(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.p_global);
            vkCmdBindDescriptorSets(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pl_global,
                                    0, 1, &sets[2], 0, nullptr);
            vkCmdPushConstants(g_vk.cmd, g_vk.pl_global, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(gp), &gp);
            vkCmdDispatch(g_vk.cmd, 1, 1, 1);
            full_barrier(g_vk.cmd);
        }

        vkEndCommandBuffer(g_vk.cmd);

        vkResetFences(g_vk.dev, 1, &g_vk.fence);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &g_vk.cmd;
        if (vkQueueSubmit(g_vk.queue, 1, &si, g_vk.fence) != VK_SUCCESS) goto cleanup;
        if (vkWaitForFences(g_vk.dev, 1, &g_vk.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) goto cleanup;
        rc = 0;
    }

cleanup:
    vk_free_buffer(&d_k);
    vk_free_buffer(&d_key);
    vk_free_buffer(&d_off);
    vk_free_buffer(&d_raw);
    vk_free_buffer(&d_agg);
    return rc;
}

/* ── ABI: compact_rows (NOT called by the runtime; KV compaction uses a CPU
 *    path in llama-kv-cache.cpp). Provided for ABI completeness; the data is a
 *    host pointer in this fork, so a host-side gather is correct. ───────── */
extern "C" int tria_vk_compact_rows(
        void * tensor_data,
        const uint32_t * h_indices,
        uint32_t n_keep,
        uint32_t first_move,
        uint32_t row_bytes) {
    if (!tensor_data || !h_indices) return -1;
    if (first_move >= n_keep || row_bytes == 0) return 0;
    const uint32_t n_move = n_keep - first_move;
    uint8_t * base = static_cast<uint8_t*>(tensor_data);
    std::vector<uint8_t> scratch((size_t)n_move * row_bytes);
    for (uint32_t i = 0; i < n_move; i++) {
        const uint32_t src = h_indices[first_move + i];
        memcpy(scratch.data() + (size_t)i * row_bytes,
               base + (size_t)src * row_bytes, row_bytes);
    }
    memcpy(base + (size_t)first_move * row_bytes, scratch.data(),
           (size_t)n_move * row_bytes);
    return 0;
}
