#pragma once

#include "llama-graph.h"
#include "llama-kv-cache-paged.h"
#include "llama-memory-recurrent.h"

// Build the two inputs of a hybrid-paged model independently. The graph does
// not need a new composite input type.
llm_graph_input_attn_kv_paged * llama_hybrid_paged_build_attn_inp(
        llm_graph_context & gctx,
        const llama_kv_cache_paged_context * paged_mctx);

llm_graph_input_rs * llama_hybrid_paged_build_rs_inp(
        llm_graph_context & gctx,
        const llama_memory_recurrent_context * recr_mctx);
