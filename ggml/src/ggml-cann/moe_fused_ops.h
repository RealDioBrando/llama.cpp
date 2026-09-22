#ifndef CANN_MOE_FUSED_OPS_H
#define CANN_MOE_FUSED_OPS_H

#include "common.h"

struct ggml_cgraph;
struct ggml_tensor;

bool ggml_cann_moe_router_fused(ggml_backend_cann_context & ctx,
                                const ggml_cgraph * graph, int node_idx,
                                int * last_fused_node);
bool ggml_cann_moe_gate_up_fused(ggml_backend_cann_context & ctx,
                                 const ggml_cgraph * graph, int node_idx,
                                 int * last_fused_node);
bool ggml_cann_moe_down_unpermute_fused(ggml_backend_cann_context & ctx,
                                        const ggml_cgraph * graph, int node_idx,
                                        int * last_fused_node);

#endif
