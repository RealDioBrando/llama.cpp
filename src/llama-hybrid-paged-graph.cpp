#include "llama-hybrid-paged-graph.h"

#include <memory>

llm_graph_input_attn_kv_paged * llama_hybrid_paged_build_attn_inp(
        llm_graph_context & gctx,
        const llama_kv_cache_paged_context * paged_mctx) {
    GGML_ASSERT(paged_mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_paged>(
            gctx.hparams, gctx.cparams, paged_mctx);

    const int32_t n_tokens   = paged_mctx->get_n_tokens();
    const int32_t batch_size = paged_mctx->get_batch_size();
    const int32_t max_blocks = paged_mctx->get_max_blocks();

    inp->paged_write_slots   = ggml_new_tensor_1d(gctx.ctx0, GGML_TYPE_I32, n_tokens);
    inp->paged_block_table   = ggml_new_tensor_2d(gctx.ctx0, GGML_TYPE_I32, max_blocks, batch_size);
    inp->paged_context_lens  = ggml_new_tensor_1d(gctx.ctx0, GGML_TYPE_I32, batch_size);
    inp->paged_batch_offsets = ggml_new_tensor_1d(gctx.ctx0, GGML_TYPE_I32, batch_size);
    inp->paged_batch_lens    = ggml_new_tensor_1d(gctx.ctx0, GGML_TYPE_I32, batch_size);

    ggml_set_name(inp->paged_block_table, "paged_block_table");
    ggml_set_name(inp->paged_context_lens, "paged_context_lens");
    ggml_set_name(inp->paged_batch_offsets, "paged_batch_offsets");
    ggml_set_name(inp->paged_batch_lens, "paged_batch_lens");

    ggml_set_input(inp->paged_write_slots);
    ggml_set_input(inp->paged_block_table);
    ggml_set_input(inp->paged_context_lens);
    ggml_set_input(inp->paged_batch_offsets);
    ggml_set_input(inp->paged_batch_lens);

    return static_cast<llm_graph_input_attn_kv_paged *>(gctx.res->add_input(std::move(inp)));
}

llm_graph_input_rs * llama_hybrid_paged_build_rs_inp(
        llm_graph_context & gctx,
        const llama_memory_recurrent_context * recr_mctx) {
    GGML_ASSERT(recr_mctx);

    auto inp = std::make_unique<llm_graph_input_rs>(recr_mctx);

    const int64_t n_rs   = recr_mctx->get_n_rs();
    const int64_t n_seqs = gctx.ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(gctx.ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);

    inp->s_copy_main  = ggml_view_1d(gctx.ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_extra = ggml_view_1d(
            gctx.ctx0, inp->s_copy, n_rs - n_seqs, n_seqs * inp->s_copy->nb[0]);

    inp->head = recr_mctx->get_head();
    inp->rs_z = recr_mctx->get_rs_z();

    return static_cast<llm_graph_input_rs *>(gctx.res->add_input(std::move(inp)));
}
