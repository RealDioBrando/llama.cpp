#include "llama-memory-hybrid-paged.h"

#include "ggml-backend.h"
#include "llama-arch.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <stdexcept>

//
// llama_memory_hybrid_paged
//

llama_memory_hybrid_paged::llama_memory_hybrid_paged(
        const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                        uint32_t   block_size,
                        uint32_t   n_gpu_blocks,
                        uint32_t   n_cpu_blocks,
                            float   watermark,
                    ggml_type   type_r,
                    ggml_type   type_s,
                        uint32_t   rs_size,
                        uint32_t   n_ubatch,
                        uint32_t   n_seq_max,
                        uint32_t   n_rs_seq,
                            bool   offload) :
    mem_attn(new llama_kv_cache_paged(
            model.hparams.n_embd_head_v(),
            model.hparams.n_head_kv(),
            block_size,
            model.hparams.n_layer(),
            n_ubatch,
            n_seq_max)),
    mem_recr(new llama_memory_recurrent(
            model,
            type_r,
            type_s,
            offload,
            rs_size,
            n_seq_max,
            n_rs_seq,
            [&](int32_t il) { return il >= 0 && model.hparams.is_recr(il); })) {
    GGML_ASSERT(type_k == type_v && "paged cache requires matching K and V types");
    GGML_ASSERT(n_gpu_blocks > 0 && n_cpu_blocks > 0);

    auto * gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu_dev && "no CPU device found for paged KV cache");

    auto * backend_gpu = gpu_dev ? ggml_backend_dev_init(gpu_dev, nullptr)
                                 : ggml_backend_dev_init(cpu_dev, nullptr);
    auto * backend_cpu = ggml_backend_dev_init(cpu_dev, nullptr);

    mem_attn->init(backend_gpu, backend_cpu, type_k, n_gpu_blocks, n_cpu_blocks, watermark);
}

llama_memory_context_ptr llama_memory_hybrid_paged::init_batch(
        llama_batch_allocr & balloc,
        uint32_t             n_ubatch,
        bool                 embd_all) {
    balloc.split_reset();

    std::vector<llama_ubatch> ubatches;
    while (true) {
        llama_ubatch ubatch;

        if (embd_all) {
            ubatch = balloc.split_seq(n_ubatch);
        } else {
            // Keep the trailing rollback snapshots of each sequence together.
            ubatch = balloc.split_equal(
                    n_ubatch,
                    true,
                    mem_recr->n_rs_seq > 0 ? mem_recr->n_rs_seq + 1 : 0);
        }

        if (ubatch.n_tokens == 0) {
            break;
        }

        ubatches.push_back(std::move(ubatch));
    }

    if (balloc.get_n_used() < balloc.get_n_tokens()) {
        return std::make_unique<llama_memory_hybrid_paged_context>(
                LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    if (!mem_recr->prepare(ubatches)) {
        LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
        return std::make_unique<llama_memory_hybrid_paged_context>(
                LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    auto ctx_attn = mem_attn->create_batch_context(ubatches);
    if (!ctx_attn || ctx_attn->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to prepare paged ubatches\n", __func__);
        return std::make_unique<llama_memory_hybrid_paged_context>(
                LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    auto ctx_recr = std::make_unique<llama_memory_recurrent_context>(
            mem_recr.get(), ubatches);

    return std::make_unique<llama_memory_hybrid_paged_context>(
            this,
            std::move(ubatches),
            std::move(ctx_attn),
            std::move(ctx_recr));
}

llama_memory_context_ptr llama_memory_hybrid_paged::init_full() {
    return std::make_unique<llama_memory_hybrid_paged_context>(
            this,
            std::vector<llama_ubatch>{},
            mem_attn->init_full(),
            mem_recr->init_full());
}

llama_memory_context_ptr llama_memory_hybrid_paged::init_update(
        llama_context * lctx,
        bool            optimize) {
    return std::make_unique<llama_memory_hybrid_paged_context>(
            this,
            std::vector<llama_ubatch>{},
            mem_attn->init_update(lctx, optimize),
            mem_recr->init_update(lctx, optimize));
}

bool llama_memory_hybrid_paged::get_can_shift() const {
    return false;
}

void llama_memory_hybrid_paged::clear(bool data) {
    mem_attn->clear(data);
    mem_recr->clear(data);
}

bool llama_memory_hybrid_paged::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    return mem_attn->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_paged::seq_cp(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos    p0,
        llama_pos    p1) {
    // The paged scheduler drives attention-side block sharing via
    // llama_kv_cache_paged::share_blocks; seq_cp only forks the GDN
    // state (used by prefix-cache checkpoint slots).
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid_paged::seq_keep(llama_seq_id seq_id) {
    mem_recr->seq_keep(seq_id);
}

uint32_t llama_memory_hybrid_paged::get_n_seq_max() const {
    return mem_recr->get_n_seq_max();
}

void llama_memory_hybrid_paged::seq_add(
        llama_seq_id seq_id,
        llama_pos    p0,
        llama_pos    p1,
        llama_pos    shift) {
    GGML_UNUSED(seq_id);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    GGML_UNUSED(shift);

    throw std::runtime_error("seq_add is not supported by hybrid-paged memory");
}

void llama_memory_hybrid_paged::seq_div(
        llama_seq_id seq_id,
        llama_pos    p0,
        llama_pos    p1,
        int          d) {
    GGML_UNUSED(seq_id);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    GGML_UNUSED(d);

    throw std::runtime_error("seq_div is not supported by hybrid-paged memory");
}

llama_pos llama_memory_hybrid_paged::seq_pos_min(llama_seq_id seq_id) const {
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid_paged::seq_pos_max(llama_seq_id seq_id) const {
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_paged::memory_breakdown() const {
    auto breakdown = mem_attn->memory_breakdown();

    for (const auto & [buft, size] : mem_recr->memory_breakdown()) {
        breakdown[buft] += size;
    }

    return breakdown;
}

void llama_memory_hybrid_paged::state_write(
        llama_io_write_i &    io,
        llama_seq_id          seq_id,
        llama_state_seq_flags flags) const {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);

    throw std::runtime_error("state save is not supported by hybrid-paged memory");
}

void llama_memory_hybrid_paged::state_read(
        llama_io_read_i &     io,
        llama_seq_id          seq_id,
        llama_state_seq_flags flags) {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);

    throw std::runtime_error("state load is not supported by hybrid-paged memory");
}

//
// llama_memory_hybrid_paged_context
//

llama_memory_hybrid_paged_context::llama_memory_hybrid_paged_context(
        llama_memory_status status) : status(status) {}

llama_memory_hybrid_paged_context::llama_memory_hybrid_paged_context(
        llama_memory_hybrid_paged * mem,
        std::vector<llama_ubatch>   ubatches,
        llama_memory_context_ptr    ctx_attn,
        llama_memory_context_ptr    ctx_recr) :
    mem(mem),
    ubatches(std::move(ubatches)),
    ctx_attn(std::move(ctx_attn)),
    ctx_recr(std::move(ctx_recr)),
    status(llama_memory_status_combine(this->ctx_attn->get_status(),
                                       this->ctx_recr->get_status())) {
}

bool llama_memory_hybrid_paged_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    return ++i_next < ubatches.size();
}

bool llama_memory_hybrid_paged_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = ctx_attn->apply();
    res &= ctx_recr->apply();

    return res;
}

llama_memory_status llama_memory_hybrid_paged_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_paged_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_paged_context * llama_memory_hybrid_paged_context::get_paged_attn() const {
    return static_cast<const llama_kv_cache_paged_context *>(ctx_attn.get());
}

const llama_memory_recurrent_context * llama_memory_hybrid_paged_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}

//
// paged memory factory
//

llama_memory_i * llama_paged_memory_try_create(
        const llama_model &         model,
        const llama_memory_params & params,
        const llama_cparams &       cparams) {
    const llama_hparams & hparams = model.hparams;

    if (params.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT) {
        return nullptr;
    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        return nullptr;
    }

    if (params.type_k != params.type_v) {
        return nullptr;
    }

    if (model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN35MOE) {
        return new llama_memory_hybrid_paged(
                model,
                params.type_k,
                params.type_v,
                cparams.block_size,
                cparams.n_gpu_blocks,
                cparams.n_cpu_blocks,
                cparams.kv_paged_watermark,
                GGML_TYPE_F32,
                GGML_TYPE_F32,
                std::max(1u, cparams.n_seq_max),
                cparams.n_ubatch,
                cparams.n_seq_max,
                cparams.n_rs_seq,
                cparams.offload_kqv);
    }

    return nullptr;
}
