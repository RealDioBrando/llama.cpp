#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cache-paged.h"
#include "llama-memory.h"
#include "llama-memory-recurrent.h"

#include <memory>
#include <vector>

struct llama_cparams;
struct llama_memory_params;
struct llama_model;

class llama_memory_hybrid_paged : public llama_memory_i {
public:
    llama_memory_hybrid_paged(
            const llama_model & model,
                            /* paged attention */
                    ggml_type   type_k,
                    ggml_type   type_v,
                        uint32_t   block_size,
                        uint32_t   n_gpu_blocks,
                        uint32_t   n_cpu_blocks,
                            float   watermark,
                            /* recurrent */
                    ggml_type   type_r,
                    ggml_type   type_s,
                        uint32_t   rs_size,
                            /* common */
                        uint32_t   n_ubatch,
                        uint32_t   n_seq_max,
                        uint32_t   n_rs_seq,
                            bool   offload);

    ~llama_memory_hybrid_paged() = default;

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;
    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read(llama_io_read_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    llama_kv_cache_paged * get_paged_mem() const {
        return mem_attn.get();
    }

    uint32_t get_n_seq_max() const;

private:
    const std::unique_ptr<llama_kv_cache_paged> mem_attn;
    const std::unique_ptr<llama_memory_recurrent> mem_recr;
};

class llama_memory_hybrid_paged_context : public llama_memory_context_i {
public:
    llama_memory_hybrid_paged_context(llama_memory_status status);

    llama_memory_hybrid_paged_context(
            llama_memory_hybrid_paged * mem,
            std::vector<llama_ubatch> ubatches,
            llama_memory_context_ptr ctx_attn,
            llama_memory_context_ptr ctx_recr);

    ~llama_memory_hybrid_paged_context() = default;

    bool next() override;
    bool apply() override;

    llama_memory_status get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    const llama_kv_cache_paged_context * get_paged_attn() const;
    const llama_memory_recurrent_context * get_recr() const;

private:
    llama_memory_hybrid_paged * mem;

    std::vector<llama_ubatch> ubatches;
    size_t i_next = 0;

    llama_memory_context_ptr ctx_attn;
    llama_memory_context_ptr ctx_recr;

    llama_memory_status status;
};

llama_memory_i * llama_paged_memory_try_create(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams);
