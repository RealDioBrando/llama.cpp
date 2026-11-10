#pragma once

#include "llama-batch.h"
#include "llama-block-manager.h"
#include "llama-linear-prefix-index.h"
#include "llama-memory-recurrent.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct llama_model;
struct llama_cparams;

// Sparse prefix checkpoints for linear-attention (recurrent) layers.
//
// A state block holds the complete recurrent state of every linear layer at
// one token boundary, including the (1 + n_rs_seq) rollback planes.  State
// blocks live in a GPU pool and a CPU pool, each with its own
// llama_block_manager.  A block is either:
//
//   ACTIVE  - owned by exactly one running logical sequence; writable by the graph
//   CACHED  - read-only checkpoint referenced by the prefix index
//
// Checkpoint hits copy a CACHED block into a fresh ACTIVE block; mutable
// state is never shared between sequences.
class llama_linear_state_cache {
public:
    llama_linear_state_cache(const llama_model & model,
                             ggml_type         type_r,
                             ggml_type         type_s,
                             bool              offload,
                             uint32_t          n_gpu_blocks,
                             uint32_t          n_cpu_blocks,
                             uint32_t          n_seq_max,
                             uint32_t          n_rs_seq,
                             ggml_backend_t    backend_gpu,
                             ggml_backend_t    backend_cpu);

    ~llama_linear_state_cache() = default;

    llama_memory_recurrent * get_mem() const { return mem.get(); }

    //
    // Active sequence state
    //

    bool has_active(llama_seq_id seq_id) const;
    bool is_active_on_gpu(llama_seq_id seq_id) const;
    // Allocate a zeroed ACTIVE state block on the GPU pool. Returns true if
    // the sequence already had one. `tag` is recorded as the owner tag of the
    // sequence; it is only propagated to checkpoint blocks, never to ACTIVE
    // state blocks (they are not reusable).
    bool ensure_active(llama_seq_id seq_id, const std::string & tag = "");
    bool release(llama_seq_id seq_id);

    // Record the owner tag of a sequence without touching its ACTIVE block.
    bool set_seq_tag(llama_seq_id seq_id, const std::string & tag);
    const std::string & seq_tag(llama_seq_id seq_id) const;

    // Fork the state of `src` into `dst` (copy-on-fork, Q2).
    bool seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst);

    // Rollback-capable removal. A partial range at the tail of the current
    // state selects one of the saved rollback planes for the next batch.
    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1);
    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift);
    void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d);
    void seq_keep(llama_seq_id seq_id);

    void set_seq_pos(llama_seq_id seq_id, llama_pos pos);
    llama_pos seq_pos_min(llama_seq_id seq_id) const;
    llama_pos seq_pos_max(llama_seq_id seq_id) const;

    //
    // Checkpoints
    //

    // Copy the ACTIVE state of `seq_id` into a new CACHED block and index it
    // under chain_hash at token boundary `pos`.
    bool create_checkpoint(llama_seq_id seq_id, uint64_t chain_hash, llama_pos pos);

    // Deepest checkpoint covered by the given block chain (at most max_blocks).
    const llama_linear_prefix_entry * find_checkpoint(const std::vector<uint64_t> & chain_hashes,
                                                       uint32_t                     max_blocks) const;

    // Copy the checkpoint for chain_hash into a fresh ACTIVE GPU block owned
    // by seq_id. Fails atomically when no GPU block can be obtained.
    // `tag` is the request's owner tag: on failure the sequence state is gone,
    // so a retry cannot recover it from seq_state; callers pass it explicitly.
    bool fork_checkpoint(uint64_t chain_hash, llama_seq_id seq_id, const std::string & tag = "");

    bool checkpoint_is_live(const llama_linear_prefix_entry & e) const;

    // True when the chain already owns a live checkpoint. Keeps creation
    // idempotent and avoids a transient double allocation.
    bool has_live_checkpoint(uint64_t chain_hash) const;

    // True when at least one state block beyond the active reserve is
    // available for checkpoints (doc 10.5).
    bool has_checkpoint_capacity() const;

    //
    // Swapping
    //

    // Preflight-only variants are used by the facade to make the full-KV and
    // linear-state swap a single all-or-nothing operation.
    bool can_swap_out(llama_seq_id seq_id) const;
    bool can_swap_in(llama_seq_id seq_id) const;
    bool swap_out(llama_seq_id seq_id);
    bool swap_in(llama_seq_id seq_id);

    //
    // Capacity and eviction
    //

    uint32_t n_free_gpu_blocks() const;
    uint32_t n_free_cpu_blocks() const;
    uint32_t get_num_gpu_blocks() const;
    uint32_t get_num_cpu_blocks() const;
    bool     has_free_gpu_blocks(uint32_t n) const;
    bool     has_free_cpu_blocks(uint32_t n) const;

    // GPU-cached checkpoints may be demoted to the CPU pool under GPU
    // pressure; CACHED blocks may be evicted entirely. ACTIVE blocks are
    // never touched.
    bool demote_one_checkpoint();
    // Evict one CACHED checkpoint. `placement` selects the pool: -1 = any,
    // 0 = CPU pool only, 1 = GPU pool only. Callers that need a block from a
    // specific pool must not fall back to the other one: evicting there frees
    // nothing useful and throws away checkpoints that could still be forked.
    bool evict_one_checkpoint(int placement = -1);

    //
    // Batch mapping
    //

    // Build the immutable per-ubatch state row map. Every sequence in every
    // ubatch must already have an ACTIVE block on the GPU pool. Pending
    // rollback planes are consumed here.
    bool prepare_batch(const std::vector<llama_ubatch> & ubatches,
                       std::vector<llama_recurrent_batch_rows> & out);

    // Consume the rollback planes selected by the last successful
    // prepare_batch(). Called once the batch contexts have been created, so a
    // failed init_batch does not lose the pending rollback.
    void commit_prepared_batch();

    void clear();
    void set_prefix_enabled(bool v) { checkpoint_index.set_enabled(v); }

    // Owner tag of a physical state block (empty when untagged).
    const std::string & block_tag(uint32_t block_id) const;

    // Tag aggregation: live, GPU-resident checkpoints grouped by owner tag.
    // ACTIVE state blocks and CPU checkpoints are never included.
    void collect_tag_blocks(std::unordered_map<std::string, std::unordered_set<uint32_t>> & owned) const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct seq_state {
        uint32_t  block_id = 0;
        llama_pos pos      = -1;  // last processed token position
        uint32_t  rollback = 0;   // rollback plane selected for the next batch
        std::string tag;          // owner tag of the sequence
    };

    bool allocate_gpu_block(uint32_t & block_id);
    bool allocate_cpu_block(uint32_t & block_id);
    void release_block(uint32_t block_id);

    bool demote_checkpoint(uint64_t chain_hash, llama_linear_prefix_entry e);
    bool evict_checkpoint(uint64_t chain_hash, llama_linear_prefix_entry e);

    // placement filter: -1 any, 0 CPU, 1 GPU
    const llama_linear_prefix_entry * oldest_checkpoint(int placement) const;
    bool alloc_checkpoint_block(uint32_t & block_id);

    const uint32_t n_rs_seq;
    const uint32_t n_gpu_blocks;
    const uint32_t n_cpu_blocks;
    // State blocks that must stay available for per-request ACTIVE states.
    const uint32_t n_active_reserve;

    llama_block_manager block_manager;

    std::unique_ptr<llama_memory_recurrent> mem;

    std::unordered_map<llama_seq_id, seq_state> seqs;
    llama_linear_prefix_index                  checkpoint_index;

    std::vector<llama_seq_id> rollback_to_reset;
};
