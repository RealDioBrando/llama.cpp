#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

// Device pools for the paged linear-attention state storage. When a pool is
// provided the recurrent memory switches to "paged" mode:
//   - the state tensors are split into a GPU and a CPU pool of independent
//     state blocks, each block holding (1 + n_rs_seq) rollback planes,
//   - physical blocks are addressed by the owning llama_linear_state_cache,
//     not by seq_id,
//   - the batch context consumes an immutable per-ubatch row map instead of
//     running find_slot().
struct llama_recurrent_state_pool {
    ggml_backend_t backend_gpu = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    uint32_t       n_gpu_blocks = 0;
    uint32_t       n_cpu_blocks = 0;
};

// Immutable state-row mapping for one ubatch. Rows are absolute row indices
// in the per-layer GPU state tensors, laid out as plane-major:
//
//   row(block_local, plane) = plane * n_gpu_blocks + block_local
struct llama_recurrent_batch_rows {
    uint32_t n_seqs   = 0;
    uint32_t n_planes = 1;
    // row of the state to read for sequence i (already includes the rollback plane)
    std::vector<int32_t> read_rows;   // [n_seqs]
    // row to write plane p for sequence i: write_rows[i*n_planes + p]
    std::vector<int32_t> write_rows;  // [n_seqs * n_planes]
};

class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter,
    const llama_recurrent_state_pool * state_pool = nullptr);

    ~llama_memory_recurrent() = default;

    // Paged state pool API (valid only when is_paged() is true).
    bool     is_paged() const { return paged; }
    uint32_t n_state_planes() const { return 1 + n_rs_seq; }
    uint32_t n_gpu_state_blocks() const { return n_gpu_state_blocks_; }
    uint32_t n_cpu_state_blocks() const { return n_cpu_state_blocks_; }

    ggml_tensor * get_state_r_tensor(int32_t il, bool gpu) const;
    ggml_tensor * get_state_s_tensor(int32_t il, bool gpu) const;
    ggml_tensor * get_state_p_tensor(int32_t il, bool gpu) const;

    // Copy every plane of state block src_id to dst_id. Blocks may live in
    // either pool; the copy is a synchronous host-staged transfer.
    void copy_state_block(uint32_t src_id, bool src_gpu, uint32_t dst_id, bool dst_gpu);
    void zero_state_block(uint32_t id, bool gpu);

    uint32_t n_seq_max_hint() const { return n_seq_max; }

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    uint32_t get_n_seq_max() const { return n_seq_max; }

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;
    // a second conv history that must stay replicated across devices, so it cannot share the r row
    std::vector<ggml_tensor *> p_l;

    // paged mode: separate GPU and CPU block pools
    bool paged = false;
    uint32_t n_gpu_state_blocks_ = 0;
    uint32_t n_cpu_state_blocks_ = 0;
    std::vector<ggml_tensor *> r_l_gpu;
    std::vector<ggml_tensor *> s_l_gpu;
    std::vector<ggml_tensor *> p_l_gpu;
    std::vector<ggml_tensor *> r_l_cpu;
    std::vector<ggml_tensor *> s_l_cpu;
    std::vector<ggml_tensor *> p_l_cpu;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;
    size_t size_p_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    // paged mode: batch context driven by an immutable per-ubatch row map
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches,
            std::vector<llama_recurrent_batch_rows> batch_rows);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    bool     is_paged() const { return mem->is_paged(); }
    uint32_t get_n_planes() const { return mem->n_state_planes(); }

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;
    ggml_tensor * get_p_l(int32_t il) const;

    int32_t s_copy(int i) const;

    // paged mode: write row for plane `plane` of batch sequence `i`
    int32_t s_write(int i, uint32_t plane) const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    // paged mode row maps, one per ubatch
    std::vector<llama_recurrent_batch_rows> batch_rows;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
