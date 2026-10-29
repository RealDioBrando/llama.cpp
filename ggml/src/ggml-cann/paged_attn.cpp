#include "paged_attn.h"

#include "acl_tensor.h"
#include "aclnn_ops.h"

#include <atb/atb_infer.h>
#include <atb/operation.h>
#include <atb/utils.h>

#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_copy.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

#define ATB_CHECK(x)                                                                                  \
    do {                                                                                              \
        atb::Status _st = (x);                                                                        \
        if (_st != atb::NO_ERROR) {                                                                   \
            GGML_LOG_ERROR("[CANN paged] ATB error %d at %s:%d\n", (int) _st, __FILE__, __LINE__);    \
            GGML_ABORT("ATB error: " #x);                                                             \
        }                                                                                             \
    } while (0)

struct atb_tensor_maker {
    static atb::Tensor make(void * data, const std::vector<int64_t> & shape, aclDataType dtype,
                            aclFormat format = ACL_FORMAT_ND, void * host_data = nullptr) {
        atb::Tensor t;
        t.desc.dtype  = dtype;
        t.desc.format = format;
        t.desc.shape.dimNum = (uint64_t) shape.size();
        for (size_t i = 0; i < shape.size(); i++) {
            t.desc.shape.dims[i] = shape[i];
        }
        t.deviceData = data;
        t.hostData   = host_data;
        t.dataSize   = atb::Utils::GetTensorSize(t);
        return t;
    }
};

// Global ATB state. llama.cpp dispatches ops from a single thread, the mutex
// only guards Setup against any future multi-thread dispatch.
struct atb_paged_state {
    atb::Context *   ctx       = nullptr;
    atb::Operation * rac       = nullptr;  // ReshapeCacheOperation
    atb::Operation * sa        = nullptr;  // SelfAttention (uncached prefill)
    atb::Operation * pa        = nullptr;  // PagedAttention (decode)
    atb::Operation * sf        = nullptr;  // PagedAttention (multi-token query)
    int64_t pa_heads    = -1;
    int64_t pa_kv_heads = -1;
    int64_t pa_head_dim = -1;
    int64_t sf_heads    = -1;
    int64_t sf_kv_heads = -1;
    int64_t sf_head_dim = -1;
    void *           ws_rac    = nullptr;
    uint64_t         ws_rac_sz = 0;
    void *           ws_sa     = nullptr;
    uint64_t         ws_sa_sz  = 0;
    void *           ws_pa     = nullptr;
    uint64_t         ws_pa_sz  = 0;
    void *           ws_sf     = nullptr;
    uint64_t         ws_sf_sz  = 0;
    int64_t          rac_tokens = -1;
    int64_t          sa_heads   = -1;
    int64_t          sa_kv_heads = -1;
    int64_t          sa_head_dim = -1;
    float            sa_scale    = 0.0f;
    void *           prefill_mask = nullptr;
    int64_t          prefill_mask_len = -1;
    aclrtContext     saved_ctx  = nullptr;
    std::mutex       mtx;

    ~atb_paged_state() {
        if (ctx) {
            ACL_CHECK(aclrtSetCurrentContext(saved_ctx));
            ACL_CHECK(aclrtSynchronizeDevice());
        }
        if (rac)  atb::DestroyOperation((atb::Operation *) rac);
        if (sa)   atb::DestroyOperation((atb::Operation *) sa);
        if (pa)   atb::DestroyOperation((atb::Operation *) pa);
        if (sf)   atb::DestroyOperation((atb::Operation *) sf);
        if (ctx)  atb::DestroyContext(ctx);
        if (ws_rac) aclrtFree(ws_rac);
        if (ws_sa)  aclrtFree(ws_sa);
        if (ws_pa)  aclrtFree(ws_pa);
        if (ws_sf)  aclrtFree(ws_sf);
        if (prefill_mask) aclrtFree(prefill_mask);
    }
};

atb_paged_state g_atb;

void ensure_atb_ctx(ggml_backend_cann_context & cann_ctx) {
    if (g_atb.ctx) {
        ((atb::Context *) g_atb.ctx)->SetExecuteStream(cann_ctx.stream());
        return;
    }
    atb::Context * atb_ctx = nullptr;
    ATB_CHECK(atb::CreateContext(&atb_ctx));
    ATB_CHECK(atb_ctx->SetExecuteStream(cann_ctx.stream()));
    g_atb.ctx = atb_ctx;
    aclrtGetCurrentContext(&g_atb.saved_ctx);
}

atb::Operation * create_pa_op(int num_heads, int num_kv_heads, int head_dim, bool spec) {
    atb::infer::PagedAttentionParam param;
    param.headNum             = num_heads;
    param.kvHeadNum           = num_kv_heads;
    param.qkScale             = 1.0f / std::sqrt((float) head_dim);
    param.maskType            = spec ? atb::infer::PagedAttentionParam::MASK_TYPE_SPEC
                                     : atb::infer::PagedAttentionParam::UNDEFINED;
    param.batchRunStatusEnable = false;
    param.quantType           = atb::infer::PagedAttentionParam::TYPE_QUANT_UNDEFINED;
    param.outDataType         = ACL_DT_UNDEFINED;
    param.inputLayout         = atb::infer::TYPE_BSND;
    param.scaleType           = atb::infer::PagedAttentionParam::SCALE_TYPE_TOR;
    param.mlaVHeadSize        = 0;
    param.hasQuantOffset      = false;
    param.compressType        = atb::infer::PagedAttentionParam::COMPRESS_TYPE_UNDEFINED;
    param.calcType            = spec ? atb::infer::PagedAttentionParam::CALC_TYPE_SPEC
                                     : atb::infer::PagedAttentionParam::CALC_TYPE_UNDEFINED;

    atb::Operation * op = nullptr;
    ATB_CHECK(atb::CreateOperation(param, &op));
    return op;
}

atb::Operation * create_sa_op(int num_heads, int num_kv_heads, int head_dim, float scale) {
    atb::infer::SelfAttentionParam param;
    param.calcType             = atb::infer::SelfAttentionParam::PA_ENCODER;
    param.kernelType           = atb::infer::SelfAttentionParam::KERNELTYPE_DEFAULT;
    param.clampType            = atb::infer::SelfAttentionParam::CLAMP_TYPE_UNDEFINED;
    param.maskType             = atb::infer::SelfAttentionParam::MASK_TYPE_NORM;
    param.kvcacheCfg           = atb::infer::SelfAttentionParam::K_CACHE_V_CACHE;
    param.scaleType            = atb::infer::SelfAttentionParam::SCALE_TYPE_TOR;
    param.quantType            = atb::infer::SelfAttentionParam::TYPE_QUANT_UNDEFINED;
    param.cacheType            = atb::infer::SelfAttentionParam::CACHE_TYPE_NORM;
    param.outDataType          = ACL_DT_UNDEFINED;
    param.headNum              = num_heads;
    param.kvHeadNum            = num_kv_heads;
    param.qScale               = 1.0f;
    param.qkScale              = scale;
    param.batchRunStatusEnable = false;
    param.isTriuMask           = 0;
    param.inputLayout          = atb::infer::TYPE_BSND;
    param.mlaVHeadSize         = 0;
    param.windowSize           = 0;

    atb::Operation * op = nullptr;
    ATB_CHECK(atb::CreateOperation(param, &op));
    return op;
}

void setup_op(atb::Operation * op, const atb::VariantPack & vp, void ** ws, uint64_t * ws_sz,
              atb::Context * atb_ctx, aclrtStream stream) {
    uint64_t needed = 0;
    {
        std::lock_guard<std::mutex> lk(g_atb.mtx);
        ATB_CHECK(op->Setup(vp, needed, atb_ctx));
    }
    const bool resize_ws = needed > *ws_sz;
    if (resize_ws) {
        aclrtSynchronizeStream(stream);
    }
    if (resize_ws) {
        if (*ws) {
            aclrtFree(*ws);
            *ws = nullptr;
            *ws_sz = 0;
        }
        aclError ret = aclrtMalloc(ws, needed, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            ret = aclrtMalloc(ws, needed, ACL_MEM_MALLOC_NORMAL_ONLY);
        }
        ACL_CHECK(ret);
        *ws_sz = needed;
    }
}

// The full-prompt causal mask only depends on the padded prompt length, so it
// can be reused by every layer and by later requests with the same length.
void * get_prefill_causal_mask(int64_t max_len, aclrtStream stream) {
    GGML_ASSERT(max_len > 0);
    if (g_atb.prefill_mask && g_atb.prefill_mask_len == max_len) {
        return g_atb.prefill_mask;
    }

    if (g_atb.prefill_mask) {
        aclrtSynchronizeStream(stream);
        ACL_CHECK(aclrtFree(g_atb.prefill_mask));
        g_atb.prefill_mask     = nullptr;
        g_atb.prefill_mask_len = -1;
    }

    const size_t mask_elems = (size_t) max_len * (size_t) max_len;
    const size_t mask_bytes = mask_elems * sizeof(uint16_t);
    void *       mask_dev   = nullptr;
    ACL_CHECK(aclrtMalloc(&mask_dev, mask_bytes, ACL_MEM_MALLOC_HUGE_FIRST));

    std::vector<uint16_t> mask_host(mask_elems);
    const float    neg_big = -10000.0f;
    const ggml_fp16_t neg_fp16 = ggml_fp32_to_fp16(neg_big);
    uint16_t neg_bits;
    memcpy(&neg_bits, &neg_fp16, sizeof(neg_bits));
    mask_host.assign(mask_elems, neg_bits);
    for (int64_t row = 0; row < max_len; ++row) {
        for (int64_t col = 0; col <= row; ++col) {
            mask_host[(size_t) row * max_len + col] = 0;
        }
    }

    ACL_CHECK(aclrtMemcpyAsync(mask_dev, mask_bytes, mask_host.data(), mask_bytes,
                               ACL_MEMCPY_HOST_TO_DEVICE, stream));
    g_atb.prefill_mask     = mask_dev;
    g_atb.prefill_mask_len = max_len;
    return mask_dev;
}

// Pool wrap: ggml K/V pool [num_blocks, n_heads_kv, block_size, head_dim] f16
// interpreted as FRACTAL_NZ [num_blocks, kv_size/16, block_size, 16].
atb::Tensor wrap_pool_nz(const ggml_tensor * pool, int num_heads_kv, int head_dim, int block_size) {
    const int64_t kv_size = (int64_t) num_heads_kv * head_dim;
    return atb_tensor_maker::make(pool->data,
                                  {pool->ne[3], kv_size / 16, block_size, 16},
                                  ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);
}

// Cast + permute ggml [D, N, T] f32 into contiguous f16 [T, N, D].
void cast_permute_to_f16_tnd(ggml_backend_cann_context & ctx, const ggml_tensor * src, void * dst_buf,
                              int64_t n_tokens, int64_t n_heads, int64_t head_dim) {
    // ggml_cann_create_tensor reverses dims, so a ggml-order ne/nb pair yields
    // an ACL view of shape [T, N, D] with element (t, n, d) = src(d, n, t).
    int64_t  src_ne[3]  = { head_dim, n_heads, n_tokens };
    size_t   src_nb[3]  = { src->nb[0], src->nb[1], src->nb[2] };
    acl_tensor_ptr acl_src = ggml_cann_create_tensor((void *) src->data, ACL_FLOAT, sizeof(float), src_ne, src_nb, 3);

    int64_t  dst_ne[3]  = { head_dim, n_heads, n_tokens };
    size_t   dst_nb[3]  = { sizeof(uint16_t), (size_t) head_dim * sizeof(uint16_t),
                            (size_t) head_dim * n_heads * sizeof(uint16_t) };
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst_buf, ACL_FLOAT16, sizeof(uint16_t), dst_ne, dst_nb, 3);

    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_src.get(), ACL_FLOAT16, acl_dst.get());
}

// Cast contiguous f16 [T, N, D] back into ggml f32 [D, N, T].
void cast_back_to_dst(ggml_backend_cann_context & ctx, const void * src_buf, ggml_tensor * dst,
                      int64_t n_tokens, int64_t n_heads, int64_t head_dim) {
    int64_t  src_ne[3]  = { head_dim, n_heads, n_tokens };
    size_t   src_nb[3]  = { sizeof(uint16_t), (size_t) head_dim * sizeof(uint16_t),
                            (size_t) head_dim * n_heads * sizeof(uint16_t) };
    acl_tensor_ptr acl_src = ggml_cann_create_tensor((void *) src_buf, ACL_FLOAT16, sizeof(uint16_t), src_ne, src_nb, 3);

    int64_t  dst_ne[3]  = { head_dim, n_heads, n_tokens };
    size_t   dst_nb[3]  = { dst->nb[0], dst->nb[1], dst->nb[2] };
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor((void *) dst->data, ACL_FLOAT, sizeof(float), dst_ne, dst_nb, 3);

    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_src.get(), ACL_FLOAT, acl_dst.get());
}

// ATB SelfAttention for uncached prompt chunks. Q/K/V are packed [T,N,D] and
// 310P PA_ENCODER requires the packed token count to be 16-aligned.
void self_attention_prefill(ggml_backend_cann_context & ctx, void * q_f16, void * k_f16, void * v_f16,
                            void * out_f16, const std::vector<int32_t> & lengths,
                            const std::vector<int32_t> & offsets, int64_t n_heads, int64_t n_heads_kv,
                            int64_t head_dim, float scale, aclrtStream stream) {
    const int64_t q_size  = n_heads * head_dim;
    const int64_t kv_size = n_heads_kv * head_dim;
    const int64_t n_seq   = (int64_t) lengths.size();
    GGML_ASSERT(n_seq > 0);

    int64_t total = 0;
    int64_t max_len = 0;
    std::vector<int32_t> op_lengths(lengths);
    for (int64_t len : op_lengths) {
        total += len;
        max_len = std::max(max_len, (int64_t) len);
    }
    const int64_t aligned = ((total + 15) / 16) * 16;
    if (aligned != total) {
        op_lengths.back() += (int32_t) (aligned - total);
        max_len = std::max(max_len, (int64_t) op_lengths.back());
    }

    std::vector<int32_t> op_offsets(n_seq);
    int32_t cursor = 0;
    for (int64_t s = 0; s < n_seq; ++s) {
        op_offsets[s] = cursor;
        cursor += op_lengths[s];
    }

    void * q_attn = q_f16;
    void * k_attn = k_f16;
    void * v_attn = v_f16;
    void * o_attn = out_f16;
    ggml_cann_pool_alloc q_alloc(ctx.pool(), aligned * q_size * sizeof(uint16_t));
    ggml_cann_pool_alloc k_alloc(ctx.pool(), aligned * kv_size * sizeof(uint16_t));
    ggml_cann_pool_alloc v_alloc(ctx.pool(), aligned * kv_size * sizeof(uint16_t));
    ggml_cann_pool_alloc o_alloc(ctx.pool(), aligned * q_size * sizeof(uint16_t));
    const size_t q_pad_bytes  = aligned * q_size * sizeof(uint16_t);
    const size_t kv_pad_bytes = aligned * kv_size * sizeof(uint16_t);
    if (aligned != total) {
        q_attn = q_alloc.get();
        k_attn = k_alloc.get();
        v_attn = v_alloc.get();
        o_attn = o_alloc.get();
        ACL_CHECK(aclrtMemsetAsync(q_attn, q_pad_bytes, 0, q_pad_bytes, stream));
        ACL_CHECK(aclrtMemsetAsync(k_attn, kv_pad_bytes, 0, kv_pad_bytes, stream));
        ACL_CHECK(aclrtMemsetAsync(v_attn, kv_pad_bytes, 0, kv_pad_bytes, stream));
        for (int64_t s = 0; s < n_seq; ++s) {
            const size_t q_bytes  = lengths[s] * q_size * sizeof(uint16_t);
            const size_t kv_bytes = lengths[s] * kv_size * sizeof(uint16_t);
            ACL_CHECK(aclrtMemcpyAsync((char *) q_attn + op_offsets[s] * q_size * sizeof(uint16_t), q_bytes,
                                       (char *) q_f16 + offsets[s] * q_size * sizeof(uint16_t), q_bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
            ACL_CHECK(aclrtMemcpyAsync((char *) k_attn + op_offsets[s] * kv_size * sizeof(uint16_t), kv_bytes,
                                       (char *) k_f16 + offsets[s] * kv_size * sizeof(uint16_t), kv_bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
            ACL_CHECK(aclrtMemcpyAsync((char *) v_attn + op_offsets[s] * kv_size * sizeof(uint16_t), kv_bytes,
                                       (char *) v_f16 + offsets[s] * kv_size * sizeof(uint16_t), kv_bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
        }
    }

    void * mask_dev = get_prefill_causal_mask(max_len, stream);

    bool sa_valid = g_atb.sa && g_atb.sa_heads == n_heads && g_atb.sa_kv_heads == n_heads_kv &&
                    g_atb.sa_head_dim == head_dim && g_atb.sa_scale == scale;
    if (!sa_valid) {
        if (g_atb.sa) {
            atb::DestroyOperation((atb::Operation *) g_atb.sa);
            g_atb.sa = nullptr;
        }
        g_atb.sa = create_sa_op((int) n_heads, (int) n_heads_kv, (int) head_dim, scale);
        g_atb.sa_heads    = n_heads;
        g_atb.sa_kv_heads = n_heads_kv;
        g_atb.sa_head_dim = head_dim;
        g_atb.sa_scale    = scale;
    }

    atb::Tensor q_t  = atb_tensor_maker::make(q_attn, {aligned, n_heads, head_dim}, ACL_FLOAT16);
    atb::Tensor k_t  = atb_tensor_maker::make(k_attn, {aligned, n_heads_kv, head_dim}, ACL_FLOAT16);
    atb::Tensor v_t  = atb_tensor_maker::make(v_attn, {aligned, n_heads_kv, head_dim}, ACL_FLOAT16);
    atb::Tensor mk_t = atb_tensor_maker::make(mask_dev, {1, max_len, max_len}, ACL_FLOAT16);
    atb::Tensor sl_t = atb_tensor_maker::make(nullptr, {n_seq}, ACL_INT32, ACL_FORMAT_ND, op_lengths.data());
    atb::Tensor o_t  = atb_tensor_maker::make(o_attn, {aligned, n_heads, head_dim}, ACL_FLOAT16);

    atb::VariantPack vp;
    vp.inTensors  = { q_t, k_t, v_t, mk_t, sl_t };
    vp.outTensors = { o_t };
    setup_op(g_atb.sa, vp, &g_atb.ws_sa, &g_atb.ws_sa_sz, (atb::Context *) g_atb.ctx, stream);
    ATB_CHECK(((atb::Operation *) g_atb.sa)->Execute(vp, (uint8_t *) g_atb.ws_sa, g_atb.ws_sa_sz,
                                                     (atb::Context *) g_atb.ctx));

    if (aligned != total) {
        for (int64_t s = 0; s < n_seq; ++s) {
            const size_t out_bytes = lengths[s] * q_size * sizeof(uint16_t);
            ACL_CHECK(aclrtMemcpyAsync((char *) out_f16 + offsets[s] * q_size * sizeof(uint16_t), out_bytes,
                                       (char *) o_attn + op_offsets[s] * q_size * sizeof(uint16_t), out_bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
        }
    }
}

}  // namespace

#ifndef GGML_CANN_HAS_ATB
void ggml_cann_paged_attn(ggml_backend_cann_context &, ggml_tensor *) {
    GGML_ABORT("ggml-cann was built without ATB support; paged KV cache requires the nnal/atb package");
}

void ggml_cann_paged_attn_cleanup(ggml_backend_cann_context &) {
}
#else

void ggml_cann_paged_attn(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    static const bool paged_timing = std::getenv("GGML_CANN_PAGED_TIMING") != nullptr;
    static const bool paged_debug  = std::getenv("GGML_CANN_PAGED_DEBUG") != nullptr;
    const int64_t total_t0 = paged_timing ? ggml_time_us() : 0;
    const ggml_tensor * q             = dst->src[0];
    const ggml_tensor * k_new         = dst->src[1];
    const ggml_tensor * v_new         = dst->src[2];
    const ggml_tensor * k_pool        = dst->src[3];
    const ggml_tensor * v_pool        = dst->src[4];
    const ggml_tensor * block_table   = dst->src[5];
    const ggml_tensor * write_slots   = dst->src[6];
    const ggml_tensor * context_lens  = dst->src[7];
    const ggml_tensor * batch_offsets = dst->src[8];
    const ggml_tensor * batch_lens    = dst->src[9];

    const float * op_params_f = (const float *) (dst->op_params);
    const float   scale       = op_params_f[0];
    const int     block_size  = ((const int32_t *) (op_params_f + 1))[0];
    const int     max_blocks  = ((const int32_t *) (op_params_f + 2))[0];
    const int64_t head_dim   = q->ne[0];
    const int64_t n_heads    = q->ne[1];
    const int64_t n_tokens   = q->ne[2];
    const int64_t n_heads_kv = k_new->ne[1];
    const int64_t n_seq      = batch_lens->ne[0];
    const int64_t kv_size    = n_heads_kv * head_dim;

    // The 310P ATB PA kernel supports 64-token blocks; the CPU kernel accepts
    // any 16-token multiple.
    GGML_ASSERT(block_size % 16 == 0 && "ATB paged attention requires 16-token block multiples");
    GGML_ASSERT(k_pool->ne[0] == head_dim && k_pool->ne[1] == block_size &&
                k_pool->ne[2] == n_heads_kv && "unexpected K pool layout");
    GGML_ASSERT(kv_size % 16 == 0 && "NZ pool requires kv_size to be a multiple of 16");

    ensure_atb_ctx(ctx);
    aclrtStream stream = ctx.stream();

    ggml_cann_pool_alloc q_alloc(ctx.pool(), n_tokens * n_heads * head_dim * sizeof(uint16_t));
    void * q_f16 = q_alloc.get();
    const int64_t q_cast_t0 = paged_timing ? ggml_time_us() : 0;
    cast_permute_to_f16_tnd(ctx, q, q_f16, n_tokens, n_heads, head_dim);
    const int64_t q_cast_us = paged_timing ? ggml_time_us() - q_cast_t0 : 0;

    // ---- write new KV into the pools ----
    ggml_cann_pool_alloc k_alloc(ctx.pool(), n_tokens * n_heads_kv * head_dim * sizeof(uint16_t));
    ggml_cann_pool_alloc v_alloc(ctx.pool(), n_tokens * n_heads_kv * head_dim * sizeof(uint16_t));
    void * k_f16 = k_alloc.get();
    void * v_f16 = v_alloc.get();
    const int64_t kv_cast_t0 = paged_timing ? ggml_time_us() : 0;
    cast_permute_to_f16_tnd(ctx, k_new, k_f16, n_tokens, n_heads_kv, head_dim);
    cast_permute_to_f16_tnd(ctx, v_new, v_f16, n_tokens, n_heads_kv, head_dim);
    const int64_t kv_cast_us = paged_timing ? ggml_time_us() - kv_cast_t0 : 0;

    atb::Tensor k_in = atb_tensor_maker::make(k_f16, {n_tokens, n_heads_kv, head_dim}, ACL_FLOAT16);
    atb::Tensor v_in = atb_tensor_maker::make(v_f16, {n_tokens, n_heads_kv, head_dim}, ACL_FLOAT16);
    atb::Tensor kc   = wrap_pool_nz(k_pool, n_heads_kv, head_dim, block_size);
    atb::Tensor vc   = wrap_pool_nz(v_pool, n_heads_kv, head_dim, block_size);
    atb::Tensor sm   = atb_tensor_maker::make((void *) write_slots->data, {n_tokens}, ACL_INT32);

    {
        const int64_t rac_t0 = paged_timing ? ggml_time_us() : 0;
        atb::VariantPack vp;
        vp.inTensors  = { k_in, v_in, kc, vc, sm };
        vp.outTensors = { kc, vc };

        // ReshapeCache needs a fresh op per token count (matches vLLM-ascend 310P)
        if (g_atb.rac && g_atb.rac_tokens != n_tokens) {
            atb::DestroyOperation((atb::Operation *) g_atb.rac);
            g_atb.rac = nullptr;
        }
        if (!g_atb.rac) {
            atb::infer::ReshapeAndCacheParam param;
            param.compressType = atb::infer::ReshapeAndCacheParam::COMPRESS_TYPE_UNDEFINED;
            param.kvCacheCfg   = atb::infer::ReshapeAndCacheParam::K_CACHE_V_CACHE_NZ;
            ATB_CHECK(atb::CreateOperation(param, &g_atb.rac));
            g_atb.rac_tokens = n_tokens;
        }

        uint64_t needed = 0;
        {
            std::lock_guard<std::mutex> lk(g_atb.mtx);
            ATB_CHECK(((atb::Operation *) g_atb.rac)->Setup(vp, needed, (atb::Context *) g_atb.ctx));
        }
        if (needed > g_atb.ws_rac_sz) {
            if (g_atb.ws_rac) {
                aclrtFree(g_atb.ws_rac);
                g_atb.ws_rac    = nullptr;
                g_atb.ws_rac_sz = 0;
            }
            aclError ret = aclrtMalloc(&g_atb.ws_rac, needed, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                ret = aclrtMalloc(&g_atb.ws_rac, needed, ACL_MEM_MALLOC_NORMAL_ONLY);
            }
            ACL_CHECK(ret);
            g_atb.ws_rac_sz = needed;
        }
        ATB_CHECK(((atb::Operation *) g_atb.rac)->Execute(vp, (uint8_t *) g_atb.ws_rac, needed,
                                                          (atb::Context *) g_atb.ctx));
        const int64_t rac_us = paged_timing ? ggml_time_us() - rac_t0 : 0;
        if (paged_timing) {
            GGML_LOG_ERROR("[paged timing] rac n_tokens=%d n_seq=%d us=%lld\n", (int) n_tokens, (int) n_seq,
                           (long long) rac_us);
        }
    }

    // ---- attention ----
    ggml_cann_pool_alloc out_alloc(ctx.pool(), n_tokens * n_heads * head_dim * sizeof(uint16_t));
    void * out_f16 = out_alloc.get();

    ggml_cann_pool_alloc cl_alloc(ctx.pool(), n_seq * sizeof(int32_t));
    const int64_t meta_t0 = paged_timing ? ggml_time_us() : 0;
    // set_input keeps the scheduler's host arrays in tensor->extra. This avoids
    // a synchronous D2H read on every layer; ATB tiling needs those host values.
    const auto * cl_fast = (const int32_t *) context_lens->extra;
    const auto * bl_fast = (const int32_t *) batch_lens->extra;
    const auto * bo_fast = (const int32_t *) batch_offsets->extra;
    const bool   use_fast_meta = cl_fast && bl_fast && bo_fast;

    std::vector<int32_t> cl_host(n_seq);
    std::vector<int32_t> bl_host(n_seq);
    std::vector<int32_t> bo_host(n_seq);
    void * cl_dev = cl_alloc.get();
    if (use_fast_meta) {
        cl_host.assign(cl_fast, cl_fast + n_seq);
        bl_host.assign(bl_fast, bl_fast + n_seq);
        bo_host.assign(bo_fast, bo_fast + n_seq);
        cl_dev = const_cast<void *>(context_lens->data);
    } else {
        ACL_CHECK(aclrtMemcpyAsync(cl_alloc.get(), n_seq * sizeof(int32_t), context_lens->data,
                                   n_seq * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
        ACL_CHECK(aclrtMemcpy(cl_host.data(), cl_host.size() * sizeof(int32_t), context_lens->data,
                              n_seq * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(bl_host.data(), bl_host.size() * sizeof(int32_t), batch_lens->data,
                              n_seq * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(bo_host.data(), bo_host.size() * sizeof(int32_t), batch_offsets->data,
                              n_seq * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
    }

    ggml_cann_pool_alloc bt_alloc(ctx.pool());
    void * bt_dev = nullptr;
    const auto * bt_fast = (const int32_t *) block_table->extra;
    if (bt_fast) {
        bt_dev = const_cast<void *>(block_table->data);
    } else {
        bt_dev = bt_alloc.alloc(n_seq * max_blocks * sizeof(int32_t));
        std::vector<int32_t> bt_lin(n_seq * max_blocks, 0);
        ACL_CHECK(aclrtMemcpy(bt_lin.data(), bt_lin.size() * sizeof(int32_t), block_table->data,
                              bt_lin.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
        for (int64_t s = 0; s < n_seq; ++s) {
            const int64_t used = std::min(max_blocks, (cl_host[s] + block_size - 1) / block_size);
            for (int64_t m = used; m < max_blocks; ++m) {
                bt_lin[s * max_blocks + m] = 0;
            }
        }
        ACL_CHECK(aclrtMemcpyAsync(bt_dev, bt_lin.size() * sizeof(int32_t), bt_lin.data(),
                                   bt_lin.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE, stream));
    }
    const int64_t meta_us = paged_timing ? ggml_time_us() - meta_t0 : 0;

    const bool decode = (n_tokens == n_seq);
    if (paged_debug) {
        GGML_LOG_INFO("[CANN paged] op: %s n_tokens=%d n_seq=%d n_heads=%d n_kv=%d hd=%d bs=%d max_blocks=%d\n",
                      decode ? "PA" : "SF", (int) n_tokens, (int) n_seq, (int) n_heads, (int) n_heads_kv,
                      (int) head_dim, block_size, max_blocks);
        std::vector<int32_t> ws_host(n_tokens);
        ACL_CHECK(aclrtMemcpy(ws_host.data(), ws_host.size() * sizeof(int32_t), write_slots->data,
                              ws_host.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
        std::vector<int32_t> bt_host(n_seq * max_blocks);
        ACL_CHECK(aclrtMemcpy(bt_host.data(), bt_host.size() * sizeof(int32_t), block_table->data,
                              bt_host.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        for (int64_t s = 0; s < n_seq; ++s) {
            GGML_LOG_INFO("[CANN paged] meta seq=%d bo=%d bl=%d cl=%d ws=[%d..%d] bt0=%d\n",
                          (int) s, (int) bo_host[s], (int) bl_host[s], (int) cl_host[s],
                          (int) ws_host[bo_host[s]], (int) ws_host[bo_host[s] + bl_host[s] - 1],
                          (int) bt_host[s * max_blocks]);
        }
    }

    if (decode) {
        atb::Tensor qa  = atb_tensor_maker::make(q_f16, {n_seq, n_heads, head_dim}, ACL_FLOAT16);
        atb::Tensor btt = atb_tensor_maker::make(bt_dev, {n_seq, max_blocks}, ACL_INT32);
        atb::Tensor clt = atb_tensor_maker::make(cl_dev, {n_seq}, ACL_INT32, ACL_FORMAT_ND, cl_host.data());
        atb::Tensor out = atb_tensor_maker::make(out_f16, {n_seq, n_heads, head_dim}, ACL_FLOAT16);

        if (!g_atb.pa || g_atb.pa_heads != n_heads || g_atb.pa_kv_heads != n_heads_kv ||
                g_atb.pa_head_dim != head_dim) {
            if (g_atb.pa) {
                atb::DestroyOperation((atb::Operation *) g_atb.pa);
                g_atb.pa = nullptr;
            }
            g_atb.pa = create_pa_op(n_heads, n_heads_kv, head_dim, /*spec=*/false);
            g_atb.pa_heads    = n_heads;
            g_atb.pa_kv_heads = n_heads_kv;
            g_atb.pa_head_dim = head_dim;
        }

        atb::VariantPack vp;
        vp.inTensors  = { qa, kc, vc, btt, clt };
        vp.outTensors = { out };
        const int64_t pa_setup_t0 = paged_timing ? ggml_time_us() : 0;
        if (paged_debug) {
            GGML_LOG_INFO("[CANN paged] PA call: cl_host min/max = %d/%d\n",
                          *std::min_element(cl_host.begin(), cl_host.end()),
                          *std::max_element(cl_host.begin(), cl_host.end()));
        }
        setup_op(g_atb.pa, vp, &g_atb.ws_pa, &g_atb.ws_pa_sz, (atb::Context *) g_atb.ctx, stream);
        const int64_t pa_setup_us = paged_timing ? ggml_time_us() - pa_setup_t0 : 0;
        const int64_t pa_exec_t0 = paged_timing ? ggml_time_us() : 0;
        ATB_CHECK(((atb::Operation *) g_atb.pa)->Execute(vp, (uint8_t *) g_atb.ws_pa, g_atb.ws_pa_sz,
                                                         (atb::Context *) g_atb.ctx));
        const int64_t pa_exec_us = paged_timing ? ggml_time_us() - pa_exec_t0 : 0;
        if (paged_timing) {
            GGML_LOG_ERROR("[paged timing] pa n_seq=%d setup_us=%lld exec_us=%lld meta_us=%lld fast=%d bt=%d q_cast_us=%lld kv_cast_us=%lld\n",
                           (int) n_seq, (long long) pa_setup_us, (long long) pa_exec_us, (long long) meta_us,
                           use_fast_meta ? 1 : 0, block_table->extra ? 1 : 0,
                           (long long) q_cast_us, (long long) kv_cast_us);
        }
    } else {
        // Prefill: uncached chunks use the vLLM 310P SelfAttention path.
        // ReshapeCache above has already staged the same K/V into the NZ pools.
        // Chunks with past context need a multi-token query over the paged
        // pool, which is the Splitfuse op.
        ggml_cann_pool_alloc sl_alloc(ctx.pool(), sizeof(int32_t));

        if (!g_atb.sf || g_atb.sf_heads != n_heads || g_atb.sf_kv_heads != n_heads_kv ||
                g_atb.sf_head_dim != head_dim) {
            if (g_atb.sf) {
                atb::DestroyOperation((atb::Operation *) g_atb.sf);
                g_atb.sf = nullptr;
            }
            g_atb.sf = create_pa_op(n_heads, n_heads_kv, head_dim, /*spec=*/true);
            g_atb.sf_heads    = n_heads;
            g_atb.sf_kv_heads = n_heads_kv;
            g_atb.sf_head_dim = head_dim;
        }

        std::vector<int32_t> prefill_lengths;
        std::vector<int32_t> prefill_offsets;
        for (int64_t seq = 0; seq < n_seq; ++seq) {
            const int32_t len     = bl_host[seq];
            const int32_t offset  = bo_host[seq];
            const int32_t ctx_len = cl_host[seq];
            if (len <= 0 || ctx_len != len) {
                continue;
            }
            prefill_lengths.push_back(len);
            prefill_offsets.push_back(offset);
        }
        if (!prefill_lengths.empty()) {
            self_attention_prefill(ctx, q_f16, k_f16, v_f16, out_f16, prefill_lengths, prefill_offsets,
                                   n_heads, n_heads_kv, head_dim, scale, stream);
        }

        for (int64_t seq = 0; seq < n_seq; ++seq) {
            const int32_t len     = bl_host[seq];
            const int32_t offset  = bo_host[seq];
            const int32_t ctx_len = cl_host[seq];
            if (len <= 0) {
                continue;
            }
            const int32_t past_len = ctx_len - len;
            int32_t len_host = len;

            if (past_len == 0) {
                continue;
            }
            const int64_t sf_t0 = paged_timing ? ggml_time_us() : 0;

            // Causal mask in NZ layout: [1, mask_ms/16, tokens_pad, 16]
            const int64_t mask_ms    = ((ctx_len + 15) / 16) * 16;
            const int64_t tokens_pad = ((len + 15) / 16) * 16;
            const int64_t mask_elems = (mask_ms / 16) * tokens_pad * 16;
            if (paged_debug) {
                GGML_LOG_INFO("[CANN paged] SF call: seq=%d len=%d ctx=%d mask_ms=%d tokens_pad=%d\n",
                              (int) seq, (int) len, (int) ctx_len, (int) mask_ms, (int) tokens_pad);
            }
            ggml_cann_pool_alloc mask_alloc(ctx.pool(), mask_elems * sizeof(uint16_t));
            std::vector<uint16_t> mask_host;
            const float neg_big = -10000.0f;
            uint16_t neg_bits;
            ggml_fp16_t nf = ggml_fp32_to_fp16(neg_big);
            memcpy(&neg_bits, &nf, sizeof(uint16_t));
            mask_host.assign(mask_elems, neg_bits);
            for (int64_t t = 0; t < len; ++t) {
                const int64_t pos = past_len + t;
                for (int64_t j = 0; j < mask_ms; ++j) {
                    if (j <= pos) {
                        mask_host[(j / 16) * tokens_pad * 16 + t * 16 + (j % 16)] = 0;
                    }
                }
            }
            ACL_CHECK(aclrtMemcpyAsync(mask_alloc.get(), mask_elems * sizeof(uint16_t), mask_host.data(),
                                       mask_elems * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE, stream));

            ACL_CHECK(aclrtMemcpyAsync(sl_alloc.get(), sizeof(int32_t), &len, sizeof(int32_t),
                                       ACL_MEMCPY_HOST_TO_DEVICE, stream));

            atb::Tensor qa  = atb_tensor_maker::make((char *) q_f16 + offset * n_heads * head_dim * sizeof(uint16_t),
                                                     {len, n_heads, head_dim}, ACL_FLOAT16);
            atb::Tensor btt = atb_tensor_maker::make((char *) bt_dev + seq * max_blocks * sizeof(int32_t),
                                                     {1, max_blocks}, ACL_INT32);
            atb::Tensor clt = atb_tensor_maker::make((char *) cl_dev + seq * sizeof(int32_t), {1}, ACL_INT32,
                                                     ACL_FORMAT_ND, &cl_host[seq]);
            atb::Tensor mk  = atb_tensor_maker::make(mask_alloc.get(), {1, mask_ms / 16, tokens_pad, 16},
                                                     ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);
            atb::Tensor slt = atb_tensor_maker::make(sl_alloc.get(), {1}, ACL_INT32, ACL_FORMAT_ND, &len_host);
            atb::Tensor out = atb_tensor_maker::make((char *) out_f16 + offset * n_heads * head_dim * sizeof(uint16_t),
                                                     {len, n_heads, head_dim}, ACL_FLOAT16);

            atb::VariantPack vp;
            vp.inTensors  = { qa, kc, vc, btt, clt, mk, slt };
            vp.outTensors = { out };
            setup_op(g_atb.sf, vp, &g_atb.ws_sf, &g_atb.ws_sf_sz, (atb::Context *) g_atb.ctx, stream);
            ATB_CHECK(((atb::Operation *) g_atb.sf)->Execute(vp, (uint8_t *) g_atb.ws_sf, g_atb.ws_sf_sz,
                                                             (atb::Context *) g_atb.ctx));
            if (paged_timing) {
                GGML_LOG_ERROR("[paged timing] sf seq=%d len=%d us=%lld\n", (int) seq, (int) len,
                               (long long) (ggml_time_us() - sf_t0));
            }
        }
    }

    const int64_t cast_back_t0 = paged_timing ? ggml_time_us() : 0;
    cast_back_to_dst(ctx, out_f16, dst, n_tokens, n_heads, head_dim);
    if (paged_timing) {
        GGML_LOG_ERROR("[paged timing] total n_tokens=%d n_seq=%d us=%lld cast_back_us=%lld\n",
                       (int) n_tokens, (int) n_seq, (long long) (ggml_time_us() - total_t0),
                       (long long) (ggml_time_us() - cast_back_t0));
    }
}

void ggml_cann_paged_attn_cleanup(ggml_backend_cann_context & cann_ctx) {
    if (!g_atb.ctx) {
        return;
    }

    ACL_CHECK(aclrtSetCurrentContext(g_atb.saved_ctx));
    ((atb::Context *) g_atb.ctx)->SetExecuteStream(cann_ctx.stream());
    ACL_CHECK(aclrtSynchronizeDevice());

    if (g_atb.rac)  atb::DestroyOperation((atb::Operation *) g_atb.rac);
    if (g_atb.sa)   atb::DestroyOperation((atb::Operation *) g_atb.sa);
    if (g_atb.pa)   atb::DestroyOperation((atb::Operation *) g_atb.pa);
    if (g_atb.sf)   atb::DestroyOperation((atb::Operation *) g_atb.sf);
    atb::DestroyContext((atb::Context *) g_atb.ctx);

    if (g_atb.ws_rac) ACL_CHECK(aclrtFree(g_atb.ws_rac));
    if (g_atb.ws_sa)  ACL_CHECK(aclrtFree(g_atb.ws_sa));
    if (g_atb.ws_pa)  ACL_CHECK(aclrtFree(g_atb.ws_pa));
    if (g_atb.ws_sf)  ACL_CHECK(aclrtFree(g_atb.ws_sf));
    if (g_atb.prefill_mask) ACL_CHECK(aclrtFree(g_atb.prefill_mask));

    g_atb.ctx       = nullptr;
    g_atb.rac       = nullptr;
    g_atb.sa        = nullptr;
    g_atb.pa        = nullptr;
    g_atb.sf        = nullptr;
    g_atb.ws_rac    = nullptr;
    g_atb.ws_rac_sz = 0;
    g_atb.ws_sa     = nullptr;
    g_atb.ws_sa_sz  = 0;
    g_atb.ws_pa     = nullptr;
    g_atb.ws_pa_sz  = 0;
    g_atb.ws_sf     = nullptr;
    g_atb.ws_sf_sz  = 0;
    g_atb.rac_tokens = -1;
    g_atb.sa_heads   = -1;
    g_atb.sa_kv_heads = -1;
    g_atb.sa_head_dim = -1;
    g_atb.sa_scale    = 0.0f;
    g_atb.prefill_mask = nullptr;
    g_atb.prefill_mask_len = -1;
    g_atb.saved_ctx = nullptr;
}

#endif  // GGML_CANN_HAS_ATB
