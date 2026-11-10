#include "llama-memory-hybrid-paged.h"

#include "ggml-backend.h"
#include "llama-arch.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
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
                            bool   offload,
                        uint32_t   n_gpu_state_blocks,
                        uint32_t   n_cpu_state_blocks) :
    mem_attn(new llama_kv_cache_paged(
            model.hparams.n_embd_head_v(),
            model.hparams.n_head_kv(),
            block_size,
            model.hparams.n_layer(),
            n_ubatch,
            n_seq_max)) {
    GGML_ASSERT(type_k == type_v && "paged cache requires matching K and V types");
    GGML_ASSERT(n_gpu_blocks > 0 && n_cpu_blocks > 0 &&
                n_gpu_state_blocks > 0 && n_cpu_state_blocks > 0);
    GGML_UNUSED(rs_size);

    // use the device of the model layers so -mg pins the paged KV pools too;
    // with --split-mode tensor this is the meta device wrapping all TP ranks
    auto * gpu_dev = model.dev_layer(0);
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu_dev && "no CPU device found for paged KV cache");

    const auto gpu_dev_type = gpu_dev ? ggml_backend_dev_type(gpu_dev) : GGML_BACKEND_DEVICE_TYPE_CPU;
    auto * backend_gpu = gpu_dev_type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                                 gpu_dev_type == GGML_BACKEND_DEVICE_TYPE_META
                             ? ggml_backend_dev_init(gpu_dev, nullptr)
                             : ggml_backend_dev_init(cpu_dev, nullptr);
    auto * backend_cpu = ggml_backend_dev_init(cpu_dev, nullptr);

    // full-attention layers only: recurrent layers own no paged K/V
    mem_attn->init(backend_gpu, backend_cpu, type_k, n_gpu_blocks, n_cpu_blocks, watermark,
            [&model](uint32_t il) { return !model.hparams.is_recr(il); });

    mem_linear = std::make_unique<llama_linear_state_cache>(
            model,
            type_r,
            type_s,
            offload,
            n_gpu_state_blocks,
            n_cpu_state_blocks,
            n_seq_max,
            n_rs_seq,
            backend_gpu,
            backend_cpu);
}

llama_memory_context_ptr llama_memory_hybrid_paged::init_batch(
        llama_batch_allocr & balloc,
        uint32_t             n_ubatch,
        bool                 embd_all) {
    llama_memory_recurrent * mem_recr = mem_linear->get_mem();

    balloc.split_reset();

    std::vector<llama_ubatch> ubatches;
    while (true) {
        llama_ubatch ubatch;

        if (embd_all) {
            ubatch = balloc.split_seq(n_ubatch);
        } else {
            // Paged KV uses write_slots/block_table and the recurrent cache is
            // indexed by seq_id: no need for the non-unified-KV sequential
            // constraint. A sequential split also reorders sequences out of the
            // batch_info row order. Keep the trailing rollback snapshots of each
            // sequence together.
            ubatch = balloc.split_equal(
                    n_ubatch,
                    /*sequential=*/false,
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

    // The linear cache owns allocation and produces the immutable row map
    // consumed by the recurrent batch context.
    std::vector<llama_recurrent_batch_rows> batch_rows;
    if (!mem_linear->prepare_batch(ubatches, batch_rows)) {
        LLAMA_LOG_ERROR("%s: failed to prepare linear state ubatches\n", __func__);
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
            mem_recr, ubatches, std::move(batch_rows));

    // Batch contexts are ready; the selected rollback planes have been
    // consumed by the row map above.
    mem_linear->commit_prepared_batch();

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
            mem_linear->get_mem()->init_full());
}

llama_memory_context_ptr llama_memory_hybrid_paged::init_update(
        llama_context * lctx,
        bool            optimize) {
    return std::make_unique<llama_memory_hybrid_paged_context>(
            this,
            std::vector<llama_ubatch>{},
            mem_attn->init_update(lctx, optimize),
            mem_linear->get_mem()->init_update(lctx, optimize));
}

bool llama_memory_hybrid_paged::get_can_shift() const {
    return false;
}

void llama_memory_hybrid_paged::clear(bool data) {
    mem_attn->clear(data);
    mem_linear->clear();
}

bool llama_memory_hybrid_paged::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    const bool ok_linear = mem_linear->seq_rm(seq_id, p0, p1);
    const bool ok_attn   = mem_attn->seq_rm(seq_id, p0, p1);
    return ok_linear && ok_attn;
}

void llama_memory_hybrid_paged::seq_cp(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos    p0,
        llama_pos    p1) {
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);

    // The paged scheduler drives attention-side block sharing via
    // llama_kv_cache_paged::share_blocks; seq_cp only forks the linear state.
    mem_linear->seq_cp(seq_id_src, seq_id_dst);
}

void llama_memory_hybrid_paged::seq_keep(llama_seq_id seq_id) {
    mem_linear->seq_keep(seq_id);
}

uint32_t llama_memory_hybrid_paged::get_n_seq_max() const {
    return mem_linear->get_mem()->get_n_seq_max();
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
    return std::max(mem_attn->seq_pos_min(seq_id), mem_linear->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid_paged::seq_pos_max(llama_seq_id seq_id) const {
    return std::min(mem_attn->seq_pos_max(seq_id), mem_linear->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_paged::memory_breakdown() const {
    auto breakdown = mem_attn->memory_breakdown();

    for (const auto & [buft, size] : mem_linear->memory_breakdown()) {
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
// paged memory budget
//

llama_paged_memory_budget llama_paged_compute_budget(
        const llama_model &   model,
        const llama_cparams & cparams,
        ggml_type             type_k,
        ggml_type             type_v,
        ggml_type             type_r,
        ggml_type             type_s) {
    const llama_hparams & hp = model.hparams;

    // sample memory on the device that holds the model layers
    ggml_backend_dev_t dev = model.dev_layer(0);
    if (dev == nullptr) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    }

    size_t free  = 0;
    size_t total = 0;
    if (dev != nullptr) {
        ggml_backend_dev_memory(dev, &free, &total);
    }
    if (dev == nullptr || total == 0) {
        throw std::runtime_error(
                "paged KV: the device did not report total/free memory; "
                "the paged budget requires a queryable backend");
    }

    const uint32_t block_size  = std::max(1u, cparams.block_size);
    const size_t   type_r_size = ggml_type_size(type_r);
    const size_t   type_s_size = ggml_type_size(type_s);

    // PA block: only full-attention layers own paged K/V, and K/V use the
    // configured cache types (not the linear-state types).
    size_t pa_block_bytes = 0;
    for (uint32_t il = 0; il < hp.n_layer(); ++il) {
        if (hp.is_recr(il)) {
            continue;
        }
        pa_block_bytes += ggml_row_size(type_k, hp.n_embd_k_gqa(il));
        pa_block_bytes += ggml_row_size(type_v, hp.n_embd_v_gqa(il));
    }
    pa_block_bytes *= block_size;

    if (pa_block_bytes == 0) {
        throw std::runtime_error(
                "paged KV: model has no full-attention layers, cannot size the PA pool");
    }

    // linear state block: only recurrent layers own state, plus optional PLE conv rows
    size_t state_layer_bytes = 0;
    for (uint32_t il = 0; il < hp.n_layer(); ++il) {
        if (!hp.is_recr(il)) {
            continue;
        }
        state_layer_bytes += (size_t) hp.n_embd_r() * type_r_size;
        state_layer_bytes += (size_t) hp.n_embd_s() * type_s_size;
        if (hp.ple_conv_state() > 0 && hp.is_ple(il)) {
            state_layer_bytes += (size_t) hp.ple_conv_state() * type_r_size;
        }
    }
    const size_t state_block_bytes = (size_t) (1 + cparams.n_rs_seq) * state_layer_bytes;

    // total budget: the resident part is already outside `free`, so it is only
    // deducted once from the utilization branch (doc §10.2)
    const double utilization     = std::clamp((double) cparams.kv_paged_vram_utilization, 0.01, 1.0);
    const size_t resident_non_kv = total > free ? total - free : 0;
    const size_t util_cap        = (size_t) ((double) total * utilization);
    const size_t cache_cap       = std::min(free, util_cap > resident_non_kv ? util_cap - resident_non_kv : 0);
    const size_t runtime_non_kv  = total / 20;

    if (cache_cap <= runtime_non_kv) {
        throw std::runtime_error(
                "paged KV: no VRAM left for the paged pools after the runtime margin; "
                "total=" + std::to_string(total) +
                " free=" + std::to_string(free) +
                " resident_non_kv=" + std::to_string(resident_non_kv) +
                " runtime_non_kv=" + std::to_string(runtime_non_kv) +
                " utilization=" + std::to_string(utilization) +
                "; lower --kv-paged-vram-utilization or free VRAM");
    }

    const size_t budget = cache_cap - runtime_non_kv;

    const uint32_t n_active = std::max(1u, cparams.n_seq_max);
    const double   ratio     = std::clamp((double) cparams.kv_paged_linear_ratio, 0.0, 0.999);
    const double   watermark = std::clamp((double) cparams.kv_paged_watermark,      0.0, 0.99);

    // split the budget between the two pools.
    // 0 and the legacy common default 1 both mean "auto"; an explicit pool of
    // one block is never useful, and treating it as auto keeps the fit probe
    // and programmatic callers on the same path.
    const bool pa_explicit  = cparams.n_gpu_blocks > 1;
    const bool pa_cpu_explicit = cparams.n_cpu_blocks > 1;
    size_t linear_budget = 0;
    if (pa_explicit) {
        // explicit PA override wins; the remainder goes to linear
        const size_t pa_bytes = (size_t) cparams.n_gpu_blocks * pa_block_bytes;
        if (pa_bytes > budget) {
            throw std::runtime_error(
                    "paged KV: --n-gpu-blocks=" + std::to_string(cparams.n_gpu_blocks) +
                    " needs " + std::to_string(pa_bytes) +
                    " bytes but the paged budget is " + std::to_string(budget) +
                    " bytes; reduce --n-gpu-blocks");
        }
        linear_budget = budget - pa_bytes;
    } else if (ratio == 0.0) {
        // active-only mode: checkpoint disabled, PA gets everything else
        linear_budget = (size_t) n_active * state_block_bytes;
        if (linear_budget > budget) {
            throw std::runtime_error(
                    "paged KV: active state does not fit the paged budget; linear_budget=" +
                    std::to_string(linear_budget) + " budget=" + std::to_string(budget) +
                    "; increase --kv-paged-linear-ratio or reduce n_seq_max");
        }
    } else {
        linear_budget = (size_t) ((double) budget * ratio);
    }

    uint32_t n_gpu_state_blocks = 0;
    if (cparams.n_gpu_state_blocks > 0) {
        n_gpu_state_blocks = cparams.n_gpu_state_blocks;
    } else if (ratio == 0.0) {
        n_gpu_state_blocks = n_active;
    } else {
        n_gpu_state_blocks = state_block_bytes ? (uint32_t) (linear_budget / state_block_bytes) : 0;
    }

    uint32_t n_cpu_state_blocks = 0;
    if (cparams.n_cpu_state_blocks > 0) {
        n_cpu_state_blocks = cparams.n_cpu_state_blocks;
    } else {
        // n_gpu_state_blocks may undershoot n_active here (bad ratio or an
        // explicit override); the fail-fast below reports it. Guard the
        // subtraction so the unsigned slack cannot wrap into a huge CPU pool.
        const uint32_t ckpt_slack = n_gpu_state_blocks > n_active ? n_gpu_state_blocks - n_active : 0;
        const double   cpu_ratio  = std::min(std::max(0.0, (double) cparams.kv_paged_cpu_state_ratio), 64.0);
        n_cpu_state_blocks = n_active + (uint32_t) std::ceil((double) ckpt_slack * cpu_ratio);
    }

    // Deprecated debug cap: keep LLAMA_PAGED_PREFIX_CKPT as an upper bound on
    // budget-derived checkpoint blocks (0 disables checkpoints). Explicit pool
    // overrides win over the cap. Apply it before sizing the PA pool so the
    // freed budget is actually reusable.
    if (const char * ckpt_env = getenv("LLAMA_PAGED_PREFIX_CKPT")) {
        const int      v   = atoi(ckpt_env);
        const uint32_t cap = v > 0 ? (uint32_t) v : 0;
        LLAMA_LOG_WARN("%s: LLAMA_PAGED_PREFIX_CKPT is deprecated; "
                "capping checkpoint blocks at %u (0 disables checkpoints)\n", __func__, cap);
        if (cparams.n_gpu_state_blocks == 0) {
            const uint32_t cg = n_gpu_state_blocks > n_active ? n_gpu_state_blocks - n_active : 0;
            n_gpu_state_blocks = n_active + std::min(cg, cap);
        }
        if (cparams.n_cpu_state_blocks == 0) {
            const uint32_t cc = n_cpu_state_blocks > n_active ? n_cpu_state_blocks - n_active : 0;
            n_cpu_state_blocks = n_active + std::min(cc, cap);
        }
    }

    if (n_gpu_state_blocks < n_active) {
        const double min_ratio = (double) ((size_t) n_active * state_block_bytes) / (double) budget;
        throw std::runtime_error(
                "paged KV: linear state pool too small; n_gpu_state_blocks=" + std::to_string(n_gpu_state_blocks) +
                " state_block_bytes=" + std::to_string(state_block_bytes) +
                " n_seq_max=" + std::to_string(n_active) +
                " min_ratio=" + std::to_string(min_ratio) +
                "; increase --kv-paged-linear-ratio, --n-gpu-state-blocks, or reduce n_seq_max");
    }

    if (n_cpu_state_blocks < n_active) {
        throw std::runtime_error(
                "paged KV: CPU state pool too small; n_cpu_state_blocks=" + std::to_string(n_cpu_state_blocks) +
                " n_seq_max=" + std::to_string(n_active) +
                "; increase --n-cpu-state-blocks");
    }

    const size_t state_total = (size_t) n_gpu_state_blocks * state_block_bytes;
    if (state_total > budget) {
        throw std::runtime_error(
                "paged KV: --n-gpu-state-blocks=" + std::to_string(n_gpu_state_blocks) +
                " needs " + std::to_string(state_total) +
                " bytes but the paged budget is " + std::to_string(budget) +
                " bytes; reduce --n-gpu-state-blocks");
    }

    uint32_t n_gpu_blocks = 0;
    size_t   b_pa         = 0;
    size_t   b_ctx        = 0;
    if (pa_explicit) {
        n_gpu_blocks = cparams.n_gpu_blocks;
        b_pa         = (size_t) n_gpu_blocks * pa_block_bytes;
    } else {
        const size_t remaining = budget - state_total;
        b_pa = remaining / pa_block_bytes;
        // n_ctx is per-sequence capacity; round each sequence to whole blocks
        // first, then add the watermark headroom.
        const double ctx_blocks = std::ceil((double) cparams.n_ctx / (double) block_size);
        b_ctx = (size_t) std::ceil((double) n_active * ctx_blocks / (1.0 - watermark));
        n_gpu_blocks = (uint32_t) std::min(b_pa, b_ctx);
    }

    if (state_total + (size_t) n_gpu_blocks * pa_block_bytes > budget) {
        throw std::runtime_error(
                "paged KV: explicit pool overrides exceed the paged budget; "
                "n_gpu_state_blocks=" + std::to_string(n_gpu_state_blocks) +
                " n_gpu_blocks=" + std::to_string(n_gpu_blocks) +
                " state_block_bytes=" + std::to_string(state_block_bytes) +
                " pa_block_bytes=" + std::to_string(pa_block_bytes) +
                " budget=" + std::to_string(budget));
    }

    const uint32_t reserved = (uint32_t) std::ceil((double) n_gpu_blocks * watermark);
    const uint32_t usable   = n_gpu_blocks > reserved ? n_gpu_blocks - reserved : 0;
    if (usable < n_active) {
        throw std::runtime_error(
                "paged KV: PA pool too small; usable_gpu=" + std::to_string(usable) +
                " n_seq_max=" + std::to_string(n_active) +
                "; increase --n-gpu-blocks or reduce n_seq_max");
    }

    uint32_t n_cpu_blocks = 0;
    if (pa_cpu_explicit) {
        n_cpu_blocks = cparams.n_cpu_blocks;
    } else {
        const double cpu_ratio = std::min(std::max(0.0, (double) cparams.cpu_to_gpu_blocks_ratio), 64.0);
        n_cpu_blocks = (uint32_t) std::ceil((double) n_gpu_blocks * cpu_ratio / (1.0 - watermark));
    }

    if (n_cpu_blocks == 0) {
        throw std::runtime_error(
                "paged KV: CPU PA block pool is empty; set --n-cpu-blocks or a positive cpu_to_gpu_blocks_ratio");
    }

    const uint32_t c_gpu = n_gpu_state_blocks - n_active;
    const uint32_t c_cpu = n_cpu_state_blocks - n_active;

    if (c_gpu == 0) {
        LLAMA_LOG_WARN("paged linear: no GPU checkpoint budget (C_gpu=0); "
                "increase --kv-paged-linear-ratio or reduce n_seq_max\n");
    }

    LLAMA_LOG_INFO("paged budget: total=%zu free=%zu resident_non_kv=%zu runtime_non_kv=%zu utilization=%.2f\n",
            total, free, resident_non_kv, runtime_non_kv, utilization);
    LLAMA_LOG_INFO("paged budget: budget=%zu linear_used=%zu pa_used=%zu\n",
            budget, state_total, (size_t) n_gpu_blocks * pa_block_bytes);
    LLAMA_LOG_INFO("paged linear: block_bytes=%zu active=%u ckpt_gpu=%u ckpt_cpu=%u n_gpu=%u n_cpu=%u interval=%u base=%u\n",
            state_block_bytes, n_active, c_gpu, c_cpu, n_gpu_state_blocks, n_cpu_state_blocks,
            cparams.kv_prefix_ckpt_interval, cparams.kv_prefix_ckpt_base);
    LLAMA_LOG_INFO("paged PA: block_bytes=%zu B_pa=%zu B_ctx=%zu n_gpu=%u usable_gpu=%u n_cpu=%u watermark=%.2f\n",
            pa_block_bytes, b_pa, b_ctx, n_gpu_blocks, usable, n_cpu_blocks, watermark);
    LLAMA_LOG_INFO("paged checkpoint: interval=%u blocks base=%u max_checkpoints=%u\n",
            cparams.kv_prefix_ckpt_interval, cparams.kv_prefix_ckpt_base, c_gpu + c_cpu);

    llama_paged_memory_budget result;
    result.n_gpu_blocks       = n_gpu_blocks;
    result.n_cpu_blocks       = n_cpu_blocks;
    result.n_gpu_state_blocks = n_gpu_state_blocks;
    result.n_cpu_state_blocks = n_cpu_state_blocks;

    return result;
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

    // The paged pool uses one (head_dim, n_head_kv) pair for every layer.
    if (hparams.is_n_embd_k_gqa_variable() || hparams.is_n_embd_v_gqa_variable()) {
        throw std::runtime_error(
                "paged KV does not support per-layer varying K/V head dimensions");
    }

    if (model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN35MOE) {
        const llama_paged_memory_budget budget = llama_paged_compute_budget(
                model, cparams, params.type_k, params.type_v, GGML_TYPE_F32, GGML_TYPE_F32);

        return new llama_memory_hybrid_paged(
                model,
                params.type_k,
                params.type_v,
                cparams.block_size,
                budget.n_gpu_blocks,
                budget.n_cpu_blocks,
                cparams.kv_paged_watermark,
                GGML_TYPE_F32,
                GGML_TYPE_F32,
                std::max(1u, cparams.n_seq_max),
                cparams.n_ubatch,
                cparams.n_seq_max,
                cparams.n_rs_seq,
                cparams.offload_kqv,
                budget.n_gpu_state_blocks,
                budget.n_cpu_state_blocks);
    }

    return nullptr;
}
