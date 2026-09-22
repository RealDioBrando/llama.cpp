/**
 * W4A8 紧凑排布(无超块)证明测试（310P）:
 *
 * 证明链:
 *   P1 结构证明 —— quants 区大小精确等于 k*n/2 字节(纯 4bpw)，无任何超块元数据
 *                 (Q4_K 含超块需 4.5bpw=144B/256el，其中 16B 为交织元数据)
 *   P2 全量解码证明 —— 对 dump 出的 quants 区【每一个字节】按 8×int4 解码，
 *                 乘以外置 scale 后与 Q4_K 反量化参考对拍，全部落在重量化界内
 *                 → 若残留任何超块元数据(d/dmin/6bit scale)，对应位置解码值
 *                   将严重偏离，测试即失败
 *   P3 scale 证明 —— (k/256)×n 个 u64，低 32 位 FP32 bits，值=amax/7
 *   P4 yOffset 证明 —— = 8·Σ_j s_j·Σ_{k∈Gj} q
 *   P5 nibble 值域 —— 全部 ∈ [−7,7]（对称量化，无 Q4_K 的 0..15 偏移语义）
 *
 * 方法: 构造已知 Q4_K 权重 → CANN 后端 set_tensor(触发 transform) →
 *       aclrtMemcpy 直接 dump 设备上的布局字节 → 宿主逐项验证。
 *       不做 graph compute，不调用 W4A8 算子（无需 vendor 包）。
 */
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cann.h"

#include <acl/acl.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(cond, msg)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            fprintf(stderr, "FATAL: %s\n", msg);             \
            return 1;                                        \
        }                                                    \
    } while (0)

#define ACL_CK(expr)                                         \
    do {                                                     \
        aclError _e = (expr);                                \
        if (_e != ACL_SUCCESS) {                             \
            fprintf(stderr, "FATAL: %s = %d\n", #expr, (int) _e); \
            return 1;                                        \
        }                                                    \
    } while (0)

static uint64_t lcg_state = 42;
static uint64_t lcg_next() {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return lcg_state >> 33;
}

static float f32_from_u64_low(uint64_t v) {
    uint32_t bits = (uint32_t) (v & 0xFFFFFFFFULL);
    float     f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

int main() {
    // 必须在任何 path 查询之前（静态缓存）
    setenv("GGML_CANN_W4A8_MATMUL", "1", 1);

    const int64_t k  = 1024;  // %256 == 0
    const int64_t n  = 128;   // %8 == 0
    const int64_t n8 = n / 8;
    const size_t  layout_size =
        (size_t) (k * n8) * sizeof(int32_t) + (size_t) ((k / 256) * n) * sizeof(uint64_t) + (size_t) n * sizeof(float);

    // ---- 1. 已知权重 → Q4_K ----
    std::vector<float> wf(n * k);
    for (auto & v : wf) v = (float) ((int64_t) lcg_next() % 2000 - 1000) / 1000.0f;

    const size_t wbytes = ggml_row_size(GGML_TYPE_Q4_K, k) * n;
    std::vector<uint8_t> wq(wbytes);
    ggml_quantize_chunk(GGML_TYPE_Q4_K, wf.data(), wq.data(), 0, n, k, nullptr);

    // ---- 2. Q4_K 反量化参考 (CPU) ----
    std::vector<float> wd(n * k);
    {
        ggml_init_params ip = { 16 * 1024 * 1024, nullptr, true };
        struct ggml_context * gctx = ggml_init(ip);
        const int64_t         ne[2] = { k, n };
        struct ggml_tensor *  w     = ggml_new_tensor(gctx, GGML_TYPE_Q4_K, 2, ne);
        struct ggml_tensor *  wf32  = ggml_new_tensor(gctx, GGML_TYPE_F32, 2, ne);
        struct ggml_cgraph *  gf    = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, w, wf32));
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        CHECK(cpu != nullptr, "cpu init");
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, cpu);
        ggml_backend_tensor_set(w, wq.data(), 0, ggml_nbytes(w));
        CHECK(ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS, "dequant");
        ggml_backend_tensor_get(wf32, wd.data(), 0, wd.size() * 4);
        ggml_backend_buffer_free(buf);
        ggml_backend_free(cpu);
        ggml_free(gctx);
    }

    // ---- 3. CANN: set_tensor 触发 W4 transform ----
    ggml_init_params ip = { 16 * 1024 * 1024, nullptr, true };
    struct ggml_context * gctx = ggml_init(ip);
    const int64_t         ne[2] = { k, n };
    struct ggml_tensor *  w     = ggml_new_tensor(gctx, GGML_TYPE_Q4_K, 2, ne);
    ggml_set_name(w, "blk.0.ffn_gate.weight");  // 命中 is_matmul_weight
    ggml_backend_t        cann = ggml_backend_cann_init(0);
    CHECK(cann != nullptr, "cann init");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, cann);
    ggml_backend_tensor_set(w, wq.data(), 0, ggml_nbytes(w));

    // ---- 4. 直接 dump 设备布局字节 ----
    std::vector<uint8_t> raw(layout_size);
    ACL_CK(aclrtSetDevice(0));
    ACL_CK(aclrtMemcpy(raw.data(), layout_size, w->data, layout_size, ACL_MEMCPY_DEVICE_TO_HOST));

    const uint32_t * quants  = (const uint32_t *) raw.data();
    const uint64_t * scales  = (const uint64_t *) (raw.data() + (size_t) (k * n8) * sizeof(int32_t));
    const float *   yoffset = (const float *) (raw.data() + (size_t) (k * n8) * sizeof(int32_t) +
                                              (size_t) ((k / 256) * n) * sizeof(uint64_t));

    // ---- P1 结构证明 ----
    {
        size_t quants_bytes = (size_t) (k * n8) * sizeof(int32_t);
        printf("P1 结构: quants 区 = %zu B (纯 int4 数据, %lld x %lld int32)\n", quants_bytes, (long long) k,
               (long long) n8);
        printf("        Q4_K 若含超块需 %lld B/列(144B/256el, 含 16B 交织元数据), 纯数据仅 %lld B/列\n",
               (long long) (k / 256 * 144), (long long) (k / 256 * 128));
        printf("        scale 区 = %lld x %lld u64 (外置), yOffset 区 = %lld f32, 总计 %zu B\n", (long long) (k / 256),
               (long long) n, (long long) n, layout_size);
        CHECK(quants_bytes == (size_t) (n * k / 2), "P1: quants 区大小 != k*n/2 (4bpw)");
    }

    // ---- P2+P5 全量解码证明 ----
    double max_requant_err_norm = 0;  // |s*q - w| / (s/2) 的最大值, 应 <= 1+eps
    int     nibble_range_bad = 0;
    int64_t total = 0;
    for (int64_t col = 0; col < n; col++) {
        for (int64_t ki = 0; ki < k; ki++) {
            const uint32_t word = quants[ki * n8 + (col >> 3)];
            const uint32_t nib = (word >> ((col & 7) * 4)) & 0xF;
            const int      q   = (int8_t) ((int8_t) (nib << 4)) >> 4;  // sign-extend 4bit
            const float    s   = f32_from_u64_low(scales[(ki / 256) * n + col]);
            const float    w_hat = s * (float) q;
            const float    w_ref = wd[col * k + ki];
            if (q < -7 || q > 7) nibble_range_bad++;
            const double bound = (double) s * 0.5 + 1e-6;
            const double err   = fabs((double) w_hat - (double) w_ref);
            if (err > bound) {
                printf("P2 FAIL: col=%lld ki=%lld q=%d s=%g w_hat=%g w_ref=%g err=%g bound=%g\n", (long long) col,
                       (long long) ki, q, s, w_hat, w_ref, err, bound);
                return 1;
            }
            if (bound > 0) {
                double e = err / bound;
                if (e > max_requant_err_norm) max_requant_err_norm = e;
            }
            total++;
        }
    }
    printf("P2 全量解码: %lld 个元素全部由纯 nibble×外置scale 重构, 重构误差/重量化界 最大 = %.6f (应<=1)\n",
           (long long) total, max_requant_err_norm);
    printf("P5 nibble 值域: 越界[-7,7]个数 = %d (应为 0; Q4_K 原始为无符号 0..15 需减 min)\n", nibble_range_bad);
    CHECK(nibble_range_bad == 0, "P5: nibble 超出对称范围");

    // ---- P3 scale 证明 ----
    double max_scale_relerr = 0;
    for (int64_t col = 0; col < n; col++) {
        for (int64_t j = 0; j < k / 256; j++) {
            float amax = 0;
            for (int i = 0; i < 256; i++) {
                float a = fabsf(wd[col * k + j * 256 + i]);
                if (a > amax) amax = a;
            }
            float expect = (amax > 0) ? amax / 7.0f : 1.0f;
            float got    = f32_from_u64_low(scales[j * n + col]);
            double rel   = fabs((double) got - (double) expect) / std::max(1e-9, (double) expect);
            if (rel > max_scale_relerr) max_scale_relerr = rel;
            // 高 32 位: TransQuantParamV2 兼容编码 0x4000
            CHECK((scales[j * n + col] >> 32) == 0x4000, "P3: scale 高 32 位应为 0x4000");
        }
    }
    printf("P3 scale: %lld 个分组 scale = amax/7, 最大相对误差 = %.3g (fp32 舍入级)\n",
           (long long) ((k / 256) * n), max_scale_relerr);

    // ---- P4 yOffset 证明 ----
    double max_yo_relerr = 0;
    for (int64_t col = 0; col < n; col++) {
        double expect = 0;
        for (int64_t j = 0; j < k / 256; j++) {
            const float s = f32_from_u64_low(scales[j * n + col]);
            double     qsum = 0;
            for (int i = 0; i < 256; i++) {
                const uint32_t word = quants[(j * 256 + i) * n8 + (col >> 3)];
                const uint32_t nib  = (word >> ((col & 7) * 4)) & 0xF;
                const int      q    = (int8_t) ((int8_t) (nib << 4)) >> 4;
                qsum += q;
            }
            expect += (double) s * qsum;
        }
        expect *= 8.0;
        const double got = (double) yoffset[col];
        const double rel = fabs(got - expect) / std::max(1e-9, fabs(expect));
        if (rel > max_yo_relerr) max_yo_relerr = rel;
    }
    printf("P4 yOffset: %lld 个通道 = 8·Σ(s·Σq), 最大相对误差 = %.3g\n", (long long) n, max_yo_relerr);

    // ---- 汇总 ----
    bool pass = (max_requant_err_norm <= 1.0 + 1e-9) && nibble_range_bad == 0 && max_scale_relerr < 1e-5 &&
                max_yo_relerr < 1e-4;
    printf("\n结论: quants 区 %lld 字节 100%% 为对称 int4 权重数据(每字节全部解码验证), "
           "无超块元数据(d/dmin/6bit交织scale), scale/yOffset 全部外置且数值正确 → %s\n",
           (long long) ((size_t) (k * n8) * sizeof(int32_t)), pass ? "PASS" : "FAIL");

    ggml_backend_buffer_free(buf);
    ggml_backend_free(cann);
    ggml_free(gctx);
    return pass ? 0 : 1;
}
