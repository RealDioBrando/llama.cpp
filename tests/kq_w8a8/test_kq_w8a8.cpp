/**
 * W8A8 K-quant mul_mat 验收测试（310P）:
 *   1. 生成随机 F32 权重 → ggml_quantize_chunk 量化为 Q6_K / Q5_K
 *   2. 同一份量化字节分别喂给 CPU 后端（精确反量化参考）和 CANN 后端
 *      （加载期 W8A8 重排 + DynamicQuantV2 + aclnnQuantMatmulV5 全链路）
 *   3. 三方对比:
 *        a) NPU vs CPU-F32      —— 端到端误差（含量化噪声, 阈值松）
 *        b) NPU vs host-W8A8仿真 —— 实现正确性（同样数学, 阈值紧）
 *   4. PASS 判据: 两组 cosine 均达标
 *
 * 构建运行见同目录 run_kq_test.sh。
 */
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cann.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

#define CHECK(cond, msg)                                        \
    do {                                                        \
        if (!(cond)) {                                          \
            fprintf(stderr, "FATAL: %s\n", msg);                \
            return 1;                                           \
        }                                                       \
    } while (0)

struct CaseDef {
    const char * name;
    ggml_type    wtype;
    int64_t      k, n, m;
};

static uint64_t lcg_state = 42;
static uint64_t lcg_next() {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return lcg_state >> 33;
}

/* host 端 W8A8 仿真: 与 ggml-cann.cpp transform 相同的重量化数学 +
 * 每token int8 激活量化 + fp16 输出舍入 */
static void simulate_w8a8(const std::vector<float> & w, const std::vector<float> & x, int64_t k, int64_t n, int64_t m,
                          int gs, std::vector<float> & out) {
    // 1. weight requant (per gs-group symmetric int8)
    std::vector<int8_t>  wq(n * k);
    std::vector<float>   ws((k / gs) * n);
    for (int64_t row = 0; row < n; row++) {
        for (int64_t g = 0; g < k / gs; g++) {
            float amax = 0.0f;
            for (int i = 0; i < gs; i++) {
                float a = fabsf(w[row * k + g * gs + i]);
                if (a > amax) amax = a;
            }
            float s = (amax > 0.0f) ? amax / 127.0f : 1.0f;
            for (int i = 0; i < gs; i++) {
                int v = (int) lrintf(w[row * k + g * gs + i] / s);
                if (v > 127) v = 127;
                if (v < -127) v = -127;
                wq[row * k + g * gs + i] = (int8_t) v;
            }
            ws[g * n + row] = s;
        }
    }
    // 2. activation per-token int8
    std::vector<int8_t> xq(m * k);
    std::vector<float>  xs(m);
    for (int64_t r = 0; r < m; r++) {
        float amax = 0.0f;
        for (int64_t i = 0; i < k; i++) {
            float a = fabsf(x[r * k + i]);
            if (a > amax) amax = a;
        }
        float s = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        xs[r] = s;
        for (int64_t i = 0; i < k; i++) {
            int v = (int) lrintf(x[r * k + i] / s);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            xq[r * k + i] = (int8_t) v;
        }
    }
    // 3. matmul fp32 累加 → fp16 舍入 → fp32
    out.assign(m * n, 0.0f);
    for (int64_t r = 0; r < m; r++) {
        for (int64_t c = 0; c < n; c++) {
            double acc = 0.0;
            for (int64_t g = 0; g < k / gs; g++) {
                double gp = 0.0;
                for (int i = 0; i < gs; i++) {
                    gp += (double) xq[r * k + g * gs + i] * (double) wq[c * k + g * gs + i];
                }
                acc += gp * (double) ws[g * n + c];
            }
            acc *= (double) xs[r];
            // fp16 round
            __fp16 h = (__fp16) acc;
            out[r * n + c] = (float) h;
        }
    }
}

static bool run_case(const CaseDef & cd, int gs) {
    const int64_t k = cd.k, n = cd.n, m = cd.m;

    // ---- 1. synthetic weights & activation ----
    std::vector<float> wf(n * k), xf(m * k);
    for (auto & v : wf) v = (float) ((int64_t) lcg_next() % 2000 - 1000) / 1000.0f;
    for (auto & v : xf) v = (float) ((int64_t) lcg_next() % 2000 - 1000) / 1000.0f;

    // quantize weights to K-quant (row-major k per n-row)
    const size_t wbytes = ggml_row_size(cd.wtype, k) * n;
    std::vector<uint8_t> wq(wbytes);
    ggml_quantize_chunk(cd.wtype, wf.data(), wq.data(), 0, n, k, nullptr);

    // ---- 2. CPU reference ----
    std::vector<float> cpu_out(m * n);
    {
        ggml_init_params ip = { 32 * 1024 * 1024, nullptr, true };
        struct ggml_context * gctx = ggml_init(ip);
        const int64_t ne_w[2] = { k, n };
        const int64_t ne_x[2] = { k, m };
        struct ggml_tensor * w = ggml_new_tensor(gctx, cd.wtype, 2, ne_w);
        ggml_set_name(w, "blk.0.ffn_gate.weight");
        struct ggml_tensor * x = ggml_new_tensor(gctx, GGML_TYPE_F32, 2, ne_x);
        struct ggml_tensor * o = ggml_mul_mat(gctx, w, x);
        struct ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, o);

        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        CHECK(cpu != nullptr, "cpu backend init");
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, cpu);
        ggml_backend_tensor_set(w, wq.data(), 0, ggml_nbytes(w));
        ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));
        CHECK(ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS, "cpu compute");
        ggml_backend_tensor_get(o, cpu_out.data(), 0, cpu_out.size() * 4);
        ggml_backend_buffer_free(buf);
        ggml_backend_free(cpu);
        ggml_free(gctx);
    }

    // ---- 3. NPU (CANN) full chain ----
    std::vector<float> npu_out(m * n);
    {
        ggml_init_params ip = { 32 * 1024 * 1024, nullptr, true };
        struct ggml_context * gctx = ggml_init(ip);
        const int64_t ne_w[2] = { k, n };
        const int64_t ne_x[2] = { k, m };
        struct ggml_tensor * w = ggml_new_tensor(gctx, cd.wtype, 2, ne_w);
        ggml_set_name(w, "blk.0.ffn_gate.weight");  // must match is_matmul_weight
        struct ggml_tensor * x = ggml_new_tensor(gctx, GGML_TYPE_F32, 2, ne_x);
        struct ggml_tensor * o = ggml_mul_mat(gctx, w, x);
        struct ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, o);

        ggml_backend_t cann = ggml_backend_cann_init(0);
        CHECK(cann != nullptr, "cann init");
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, cann);
        CHECK(buf != nullptr, "cann alloc ctx tensors");
        // weight set triggers the K-quant re-quantization transform
        ggml_backend_tensor_set(w, wq.data(), 0, ggml_nbytes(w));
        ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));
        CHECK(ggml_backend_graph_compute(cann, gf) == GGML_STATUS_SUCCESS, "cann compute");
        ggml_backend_tensor_get(o, npu_out.data(), 0, npu_out.size() * 4);
        ggml_backend_buffer_free(buf);
        ggml_backend_free(cann);
        ggml_free(gctx);
    }

    // ---- 4. host W8A8 simulation ----
    // re-dequantize the K-quant weights exactly like the CPU does
    std::vector<float> wd(n * k);
    {
        ggml_init_params ip = { 32 * 1024 * 1024, nullptr, true };
        struct ggml_context * gctx = ggml_init(ip);
        const int64_t ne_w[2] = { k, n };
        struct ggml_tensor * w = ggml_new_tensor(gctx, cd.wtype, 2, ne_w);
        struct ggml_tensor * wf32 = ggml_new_tensor(gctx, GGML_TYPE_F32, 2, ne_w);
        struct ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, w, wf32));
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        CHECK(cpu != nullptr, "cpu backend init (sim)");
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, cpu);
        ggml_backend_tensor_set(w, wq.data(), 0, ggml_nbytes(w));
        CHECK(ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS, "sim dequant");
        ggml_backend_tensor_get(wf32, wd.data(), 0, wd.size() * 4);
        ggml_backend_buffer_free(buf);
        ggml_backend_free(cpu);
        ggml_free(gctx);
    }
    std::vector<float> sim_out;
    simulate_w8a8(wd, xf, k, n, m, gs, sim_out);

    // ---- 5. metrics ----
    auto metrics = [&](const std::vector<float> & a, const std::vector<float> & b) {
        double dot = 0, na = 0, nb = 0, mae = 0, mre = 0;
        for (size_t i = 0; i < a.size(); i++) {
            dot += (double) a[i] * b[i];
            na += (double) a[i] * a[i];
            nb += (double) b[i] * b[i];
            mae += fabs((double) a[i] - b[i]);
            double d = fabs((double) a[i] - b[i]) / std::max(1e-6, fabs((double) b[i]));
            if (d > mre) mre = d;
        }
        struct { double cos, mae, mre; } r{ dot / (std::sqrt(na) * std::sqrt(nb) + 1e-300), mae / a.size(), mre };
        return r;
    };
    auto r_cpu = metrics(npu_out, cpu_out);
    auto r_sim = metrics(npu_out, sim_out);

    bool pass = (r_cpu.cos >= 0.995) && (r_sim.cos >= 0.9995);
    printf("%-22s k=%-5lld n=%-5lld m=%-4lld | NPUvsCPU cos=%.6f mae=%.4f mre=%.3f | NPUvsSIM cos=%.6f mae=%.5f | %s\n",
           cd.name, (long long) k, (long long) n, (long long) m, r_cpu.cos, r_cpu.mae, r_cpu.mre, r_sim.cos, r_sim.mae,
           pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    printf("W8A8 K-quant acceptance test (gs=%d default)\n", 32);
    const int gs = 32;  // default GGML_CANN_W8A8_GROUP_SIZE
    std::vector<CaseDef> cases = {
        { "q6_k_2048x512_m33", GGML_TYPE_Q6_K, 2048, 512, 33 },
        { "q6_k_512x2048_m1",  GGML_TYPE_Q6_K, 512, 2048, 1 },
        { "q5_k_2048x2048_m8", GGML_TYPE_Q5_K, 2048, 2048, 8 },
    };
    int pass = 0, total = 0;
    for (const auto & c : cases) {
        total++;
        if (run_case(c, gs)) pass++;
    }
    printf("TOTAL: %d/%d PASS\n", pass, total);
    return (pass == total) ? 0 : 1;
}
