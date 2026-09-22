#include "moe_fused_ops.h"

#include "acl_tensor.h"
#include "aclnn_ops.h"
#include "moe_ops.h"

#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_copy.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_grouped_matmul_v4.h>
#include <aclnnop/aclnn_gather_v2.h>
#include <aclnnop/aclnn_moe_gating_top_k_softmax.h>
#include <aclnnop/aclnn_moe_gating_top_k_softmax_v2.h>
#include <aclnnop/aclnn_moe_init_routing_v2.h>
#include <aclnnop/aclnn_moe_token_unpermute.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_silu.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool env_enabled(const char * name, bool default_value) {
    return parse_bool(get_env_as_lowercase(name).value_or(default_value ? "on" : "off"));
}

const char * tensor_name(const ggml_tensor * tensor) {
    return tensor != nullptr ? ggml_get_name(tensor) : nullptr;
}

bool is_router_weight(const ggml_tensor * weight) {
    const char * name = tensor_name(weight);
    return name != nullptr && strstr(name, "ffn_gate_inp.weight") != nullptr;
}

bool is_gate_up_weight(const ggml_tensor * weight) {
    const char * name = tensor_name(weight);
    return name != nullptr && strstr(name, "ffn_gate_up_exps.weight") != nullptr;
}

bool is_gate_weight(const ggml_tensor * weight) {
    const char * name = tensor_name(weight);
    return name != nullptr && strstr(name, "ffn_gate_exps.weight") != nullptr;
}

bool is_up_weight(const ggml_tensor * weight) {
    const char * name = tensor_name(weight);
    return name != nullptr && strstr(name, "ffn_up_exps.weight") != nullptr;
}

bool is_down_weight(const ggml_tensor * weight) {
    const char * name = tensor_name(weight);
    return name != nullptr && strstr(name, "ffn_down_exps.weight") != nullptr;
}

bool is_view_like(const ggml_tensor * tensor) {
    return tensor->op == GGML_OP_RESHAPE || tensor->op == GGML_OP_VIEW ||
           tensor->op == GGML_OP_PERMUTE || tensor->op == GGML_OP_TRANSPOSE ||
           tensor->op == GGML_OP_NONE;
}

bool is_view_or_self(const ggml_tensor * tensor, const ggml_tensor * root) {
    while (tensor != nullptr && is_view_like(tensor)) {
        if (tensor == root) {
            return true;
        }
        tensor = tensor->src[0];
    }
    return tensor == root;
}

ggml_tensor * find_router_nodes(const ggml_cgraph * graph, int begin,
                                ggml_tensor ** selected_out,
                                ggml_tensor ** weights_out) {
    ggml_tensor * logits = graph->nodes[begin];
    ggml_tensor * probs = nullptr;
    ggml_tensor * sorted = nullptr;
    ggml_tensor * selected = nullptr;
    ggml_tensor * weights = nullptr;

    for (int i = begin + 1; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (node->op == GGML_OP_SOFT_MAX && node->src[0] == logits) {
            probs = node;
        } else if (probs != nullptr && sorted == nullptr && node->op == GGML_OP_ARGSORT &&
                   node->src[0] == probs) {
            sorted = node;
        } else if (sorted != nullptr && selected == nullptr && is_view_like(node) &&
                   node->src[0] == sorted && node->ne[0] < sorted->ne[0]) {
            selected = node;
        } else if (selected != nullptr && node->op == GGML_OP_RESHAPE &&
                   node->ne[0] == 1 && node->ne[1] == selected->ne[0] &&
                   node->ne[2] == selected->ne[1]) {
            weights = node;
            break;
        }
    }
    *selected_out = selected;
    *weights_out = weights;
    return probs;
}

ggml_tensor * find_glu_node(const ggml_cgraph * graph, int begin) {
    ggml_tensor * gate_up = graph->nodes[begin];
    for (int i = begin + 1; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (node->op == GGML_OP_GLU && ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU &&
                node->src[0] != nullptr && node->src[1] != nullptr &&
                is_view_or_self(node->src[0], gate_up) &&
                is_view_or_self(node->src[1], gate_up)) {
            return node;
        }
    }
    return nullptr;
}

ggml_tensor * find_glu_node_for_pair(const ggml_cgraph * graph, int begin,
                                     const ggml_tensor * gate,
                                     const ggml_tensor * up) {
    for (int i = begin + 1; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (node->op == GGML_OP_GLU && ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU &&
                node->src[0] != nullptr && node->src[1] != nullptr &&
                is_view_or_self(node->src[0], gate) &&
                is_view_or_self(node->src[1], up)) {
            return node;
        }
    }
    return nullptr;
}

ggml_tensor * find_weighted_down(const ggml_cgraph * graph, int begin,
                                 ggml_tensor ** weighted_out) {
    ggml_tensor * down = graph->nodes[begin];
    for (int i = begin + 1; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (node->op == GGML_OP_MUL && node->src[0] == down &&
                node->src[1] != nullptr && node->src[1]->ne[0] == 1 &&
                node->src[1]->ne[1] == down->ne[1] &&
                node->src[1]->ne[2] == down->ne[2]) {
            *weighted_out = node;
            return node;
        }
    }
    return nullptr;
}

ggml_tensor * find_moe_output(const ggml_cgraph * graph, int begin,
                              ggml_tensor * down,
                              ggml_tensor * weighted) {
    ggml_tensor * out = weighted;
    int end = begin;
    for (int i = begin + 1; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (is_view_like(node) && node->src[0] != nullptr &&
                (is_view_or_self(node->src[0], down) ||
                 is_view_or_self(node->src[0], weighted))) {
            end = i;
        } else if ((node->op == GGML_OP_ADD || node->op == GGML_OP_CONT) &&
                   node->src[0] != nullptr && node->src[1] != nullptr &&
                   (node->src[0] == out || is_view_or_self(node->src[0], down) ||
                    is_view_or_self(node->src[0], weighted)) &&
                   (node->src[1] == out || is_view_or_self(node->src[1], down) ||
                    is_view_or_self(node->src[1], weighted))) {
            out = node;
            end = i;
        } else if (out != nullptr && node->src[0] == out) {
            out = node;
            end = i;
        } else {
            break;
        }
    }
    (void) end;
    return out;
}

int find_node_index(const ggml_cgraph * graph, const ggml_tensor * tensor) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i] == tensor) {
            return i;
        }
    }
    return -1;
}

static void run_sorted_grouped_matmul(ggml_backend_cann_context & ctx,
                                      const ggml_tensor * weight,
                                      const void * input, const void * output,
                                      const void * cumsum_i64,
                                      int64_t n_in, int64_t n_out,
                                      int64_t n_rows, int64_t n_exp) {
    const size_t f16 = sizeof(uint16_t);
    int64_t input_ne[2] = { n_in, n_rows };
    size_t  input_nb[2] = { f16, n_in * f16 };
    int64_t weight_ne[3] = { n_out, n_in, n_exp };
    size_t  weight_nb[3] = { weight->nb[1], weight->nb[0], weight->nb[2] };
    int64_t output_ne[2] = { n_out, n_rows };
    size_t  output_nb[2] = { f16, n_out * f16 };
    int64_t cumsum_ne[1] = { n_exp };
    size_t  cumsum_nb[1] = { sizeof(int64_t) };

    const aclFormat weight_format = ggml_cann_is_weight_layout_candidate(weight)
                                        ? ACL_FORMAT_FRACTAL_NZ : ACL_FORMAT_ND;
    acl_tensor_ptr acl_input = ggml_cann_create_tensor(
        const_cast<void *>(input), ACL_FLOAT16, f16, input_ne, input_nb, 2);
    acl_tensor_ptr acl_weight = ggml_cann_create_tensor(
        weight->data, ACL_FLOAT16, f16, weight_ne, weight_nb, 3, weight_format);
    acl_tensor_ptr acl_output = ggml_cann_create_tensor(
        const_cast<void *>(output), ACL_FLOAT16, f16, output_ne, output_nb, 2);
    acl_tensor_ptr acl_cumsum = ggml_cann_create_tensor(
        const_cast<void *>(cumsum_i64), ACL_INT64, sizeof(int64_t),
        cumsum_ne, cumsum_nb, 1);

    acl_tensor_list_ptr input_list = ggml_cann_create_tensor_list(std::move(acl_input));
    acl_tensor_list_ptr weight_list = ggml_cann_create_tensor_list(std::move(acl_weight));
    acl_tensor_list_ptr output_list = ggml_cann_create_tensor_list(std::move(acl_output));
    GGML_CANN_CALL_ACLNN_OP(ctx, GroupedMatmulV4, input_list.get(), weight_list.get(),
                            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                            acl_cumsum.get(), nullptr, nullptr, nullptr, 2, 0, 0, 0,
                            output_list.get(), nullptr, nullptr);
}

static bool run_moe_full_fused(ggml_backend_cann_context & ctx,
                               const ggml_cgraph * graph, int router_idx,
                               int router_last, ggml_tensor * selected,
                               ggml_tensor * weights,
                               ggml_cann_pool_alloc * topk_ids,
                               ggml_cann_pool_alloc * topk_weights,
                               int * last_fused_node) {
    const ggml_tensor * router = graph->nodes[router_idx];
    static const bool debug =
        env_enabled("GGML_CANN_MOE_FULL_DEBUG", false);
    ggml_tensor * gate_node = nullptr;
    ggml_tensor * up_node = nullptr;
    ggml_tensor * first_expert = nullptr;
    int first_expert_idx = -1;
    for (int i = router_last + 1; i < graph->n_nodes; ++i) {
        ggml_tensor * candidate = graph->nodes[i];
        if (candidate->op == GGML_OP_MUL_MAT_ID && candidate->src[0] != nullptr &&
                (is_gate_up_weight(candidate->src[0]) || is_gate_weight(candidate->src[0]) ||
                 is_up_weight(candidate->src[0])) && candidate->src[2] == selected &&
                ggml_cann_mul_mat_id_grouped_supported(candidate)) {
            first_expert = candidate;
            first_expert_idx = i;
            break;
        }
    }
    if (first_expert == nullptr) {
        if (debug) fprintf(stderr, "[MoE FULL] fail: no expert gate/up\n");
        return false;
    }

    if (is_gate_up_weight(first_expert->src[0])) {
        gate_node = first_expert;
        up_node = first_expert;
    } else if (is_gate_weight(first_expert->src[0])) {
        gate_node = first_expert;
        for (int i = first_expert_idx + 1; i < graph->n_nodes; ++i) {
            ggml_tensor * candidate = graph->nodes[i];
            if (candidate->op == GGML_OP_MUL_MAT_ID && candidate->src[0] != nullptr &&
                    is_up_weight(candidate->src[0]) && candidate->src[2] == selected &&
                    ggml_cann_mul_mat_id_grouped_supported(candidate)) {
                up_node = candidate;
                break;
            }
        }
    } else {
        up_node = first_expert;
        for (int i = first_expert_idx + 1; i < graph->n_nodes; ++i) {
            ggml_tensor * candidate = graph->nodes[i];
            if (candidate->op == GGML_OP_MUL_MAT_ID && candidate->src[0] != nullptr &&
                    is_gate_weight(candidate->src[0]) && candidate->src[2] == selected &&
                    ggml_cann_mul_mat_id_grouped_supported(candidate)) {
                gate_node = candidate;
                break;
            }
        }
    }
    if (gate_node == nullptr || up_node == nullptr ||
            gate_node->src[1] != up_node->src[1] ||
            gate_node->ne[0] != up_node->ne[0]) {
        if (debug) fprintf(stderr, "[MoE FULL] fail: no gate/up pair\n");
        return false;
    }

    const int gate_idx = find_node_index(graph, gate_node);
    const int up_idx = find_node_index(graph, up_node);
    const int expert_begin = std::min(gate_idx, up_idx);
    ggml_tensor * glu = find_glu_node_for_pair(graph, expert_begin, gate_node, up_node);
    if (glu == nullptr || ggml_get_op_params_i32(glu, 1) != 0) {
        if (debug) fprintf(stderr, "[MoE FULL] fail: no usable glu\n");
        return false;
    }
    const int glu_idx = find_node_index(graph, glu);

    ggml_tensor * down_node = nullptr;
    for (int i = glu_idx + 1; i < graph->n_nodes; ++i) {
        ggml_tensor * candidate = graph->nodes[i];
        if (candidate->op == GGML_OP_MUL_MAT_ID && candidate->src[0] != nullptr &&
                is_down_weight(candidate->src[0]) && candidate->src[2] == selected &&
                ggml_cann_mul_mat_id_grouped_supported(candidate)) {
            down_node = candidate;
            break;
        }
    }
    if (down_node == nullptr) {
        if (debug) fprintf(stderr, "[MoE FULL] fail: no down\n");
        return false;
    }

    const int down_idx = find_node_index(graph, down_node);
    ggml_tensor * weighted = nullptr;
    if (find_weighted_down(graph, down_idx, &weighted) == nullptr ||
            weighted->src[1] != weights) {
        if (debug) fprintf(stderr, "[MoE FULL] fail: no weighted down\n");
        return false;
    }

    const int weighted_idx = find_node_index(graph, weighted);
    ggml_tensor * moe_out = find_moe_output(graph, weighted_idx, down_node, weighted);
    if (moe_out == nullptr || moe_out->type != GGML_TYPE_F32 ||
            moe_out->ne[0] != down_node->ne[0] ||
            moe_out->ne[1] != down_node->ne[2]) {
        static int output_fail_count = 0;
        if (debug && output_fail_count++ < 2) {
            fprintf(stderr, "[MoE FULL] fail: no moe_out weighted=%d null=%d\n",
                    weighted_idx, moe_out == nullptr);
            if (moe_out != nullptr) {
                fprintf(stderr,
                        "[MoE FULL]  moe type=%d ne=%ldx%ld down type=%d "
                        "w=%ld out_ne=%ldx%ldx%ld\n",
                        (int) moe_out->type, (long) moe_out->ne[0],
                        (long) moe_out->ne[1], (int) down_node->src[0]->type,
                        (long) down_node->src[0]->ne[0],
                        (long) down_node->ne[0], (long) down_node->ne[1],
                        (long) down_node->ne[2]);
            }
            for (int j = weighted_idx + 1;
                    j < std::min(graph->n_nodes, weighted_idx + 8); ++j) {
                fprintf(stderr,
                        "[MoE FULL]  node=%d op=%s name=%s ne=%ldx%ldx%ld\n",
                        j, ggml_op_name(graph->nodes[j]->op),
                        tensor_name(graph->nodes[j]), (long) graph->nodes[j]->ne[0],
                        (long) graph->nodes[j]->ne[1], (long) graph->nodes[j]->ne[2]);
            }
        }
        return false;
    }
    const int final_idx = find_node_index(graph, moe_out);
    if (glu_idx <= gate_idx || down_idx <= glu_idx || weighted_idx <= down_idx ||
            final_idx < weighted_idx) {
        if (debug) fprintf(stderr, "[MoE FULL] fail: bad node order\n");
        return false;
    }

    const int64_t n_exp    = router->src[0]->ne[1];
    const int64_t n_used   = selected->ne[0];
    const int64_t n_token  = selected->ne[1];
    const int64_t n_rows   = n_used * n_token;
    const int64_t n_hidden = gate_node->src[1]->ne[0];
    const bool merged_gate_up = gate_node == up_node;
    const int64_t n_ff     = merged_gate_up ? gate_node->ne[0] / 2 : gate_node->ne[0];
    const int64_t n_embd   = down_node->ne[0];
    if (n_ff <= 0 || n_token <= 0 || n_used <= 0 ||
            (merged_gate_up && gate_node->ne[0] % 2 != 0) ||
            gate_node->ne[0] != up_node->ne[0]) {
        if (debug) fprintf(stderr, "[MoE FULL] fail: bad shape\n");
        return false;
    }
    if (debug) {
        fprintf(stderr,
                "[MoE FULL] match E=%ld K=%ld H=%ld F=%ld T=%ld "
                "gate=%d glu=%d down=%d final=%d\n",
                (long) n_exp, (long) n_used, (long) n_hidden, (long) n_ff,
                (long) n_token, gate_idx, glu_idx, down_idx, final_idx);
    }

    const size_t f16 = sizeof(uint16_t);
    const size_t f32 = sizeof(float);
    const size_t i32 = sizeof(int32_t);

    ggml_cann_pool_alloc ids_alloc;
    ggml_cann_pool_alloc probs_alloc;
    void * ids_data = nullptr;
    void * probs_data = nullptr;
    if (topk_ids == nullptr) {
        ids_alloc.alloc(ctx.pool(), n_rows * i32);
        ids_data = ids_alloc.get();
        int64_t src_ne[2] = { n_used, n_token };
        size_t  src_nb[2] = { i32, selected->nb[1] };
        int64_t dst_ne[2] = { n_used, n_token };
        size_t  dst_nb[2] = { i32, n_used * i32 };
        acl_tensor_ptr acl_src = ggml_cann_create_tensor(
            selected->data, ACL_INT32, i32, src_ne, src_nb, 2);
        acl_tensor_ptr acl_dst = ggml_cann_create_tensor(
            ids_data, ACL_INT32, i32, dst_ne, dst_nb, 2);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_dst.get(), acl_src.get());
    } else {
        ids_data = topk_ids->get();
    }
    if (topk_weights == nullptr) {
        probs_alloc.alloc(ctx.pool(), n_rows * f32);
        probs_data = probs_alloc.get();
        int64_t dst_ne[2] = { n_used, n_token };
        size_t  dst_nb[2] = { f32, n_used * f32 };
        int64_t src_ne[2] = { n_used, n_token };
        size_t  src_nb[2] = { weights->nb[1], weights->nb[2] };
        acl_tensor_ptr acl_src = ggml_cann_create_tensor(
            weights->data, ACL_FLOAT, f32, src_ne, src_nb, 2);
        acl_tensor_ptr acl_dst = ggml_cann_create_tensor(
            probs_data, ACL_FLOAT, f32, dst_ne, dst_nb, 2);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_dst.get(), acl_src.get());
    } else {
        probs_data = topk_weights->get();
    }

    ggml_cann_pool_alloc hidden16(ctx.pool(), n_hidden * n_token * f16);
    ggml_cann_pool_alloc sorted_x(ctx.pool(), n_rows * n_hidden * f16);
    ggml_cann_pool_alloc row_idx(ctx.pool(), n_rows * i32);
    ggml_cann_pool_alloc cumsum_i32(ctx.pool(), n_exp * i32);
    ggml_cann_pool_alloc cumsum_i64(ctx.pool(), n_exp * sizeof(int64_t));
    ggml_cann_pool_alloc gate_out(ctx.pool(), n_rows * n_ff * f16);
    ggml_cann_pool_alloc up_out(ctx.pool(),
                                (merged_gate_up ? 0 : n_rows * n_ff) * f16);
    ggml_cann_pool_alloc activation(ctx.pool(), n_rows * n_ff * f16);
    ggml_cann_pool_alloc down_out(ctx.pool(), n_rows * n_embd * f16);
    ggml_cann_pool_alloc probs16(ctx.pool(), n_rows * f16);
    ggml_cann_pool_alloc routed(ctx.pool(), n_token * n_embd * f16);

    {
        const ggml_tensor * hidden = gate_node->src[1];
        int64_t src_ne[2] = { n_hidden, n_token };
        size_t  src_nb[2] = { hidden->nb[0], hidden->nb[1] };
        int64_t dst_ne[2] = { n_hidden, n_token };
        size_t  dst_nb[2] = { f16, n_hidden * f16 };
        acl_tensor_ptr acl_hidden_src = ggml_cann_create_tensor(
            hidden->data, ACL_FLOAT, f32, src_ne, src_nb, 2);
        acl_tensor_ptr acl_hidden_dst = ggml_cann_create_tensor(
            hidden16.get(), ACL_FLOAT16, f16, dst_ne, dst_nb, 2);
        GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_hidden_src.get(), ACL_FLOAT16,
                                acl_hidden_dst.get());
    }

    int64_t hidden_ne[2]  = { n_hidden, n_token };
    size_t  hidden_nb[2]  = { f16, n_hidden * f16 };
    int64_t sorted_ne[2]  = { n_hidden, n_rows };
    size_t  sorted_nb[2]  = { f16, n_hidden * f16 };
    int64_t ids_ne[2]     = { n_used, n_token };
    size_t  ids_nb[2]     = { i32, n_used * i32 };
    int64_t row_ne[1]     = { n_rows };
    size_t  row_nb[1]     = { i32 };
    int64_t cumsum_ne[1]  = { n_exp };
    size_t  c32_nb[1]     = { i32 };
    size_t  c64_nb[1]     = { sizeof(int64_t) };

    acl_tensor_ptr acl_hidden = ggml_cann_create_tensor(
        hidden16.get(), ACL_FLOAT16, f16, hidden_ne, hidden_nb, 2);
    acl_tensor_ptr acl_sorted_x = ggml_cann_create_tensor(
        sorted_x.get(), ACL_FLOAT16, f16, sorted_ne, sorted_nb, 2);
    acl_tensor_ptr acl_ids = ggml_cann_create_tensor(
        ids_data, ACL_INT32, i32, ids_ne, ids_nb, 2);
    acl_tensor_ptr acl_row_idx = ggml_cann_create_tensor(
        row_idx.get(), ACL_INT32, i32, row_ne, row_nb, 1);
    acl_tensor_ptr acl_c32 = ggml_cann_create_tensor(
        cumsum_i32.get(), ACL_INT32, i32, cumsum_ne, c32_nb, 1);
    acl_tensor_ptr acl_c64 = ggml_cann_create_tensor(
        cumsum_i64.get(), ACL_INT64, sizeof(int64_t), cumsum_ne, c64_nb, 1);
    GGML_CANN_CALL_ACLNN_OP(ctx, MoeInitRoutingV2, acl_hidden.get(), acl_ids.get(),
                            n_rows, 0, n_exp, 0, 1, false, acl_sorted_x.get(),
                            acl_row_idx.get(), acl_c32.get(), nullptr);
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_c32.get(), ACL_INT64, acl_c64.get());

    if (merged_gate_up) {
        run_sorted_grouped_matmul(ctx, gate_node->src[0], sorted_x.get(), gate_out.get(),
                                  cumsum_i64.get(), n_hidden, 2 * n_ff, n_rows, n_exp);
    } else {
        run_sorted_grouped_matmul(ctx, gate_node->src[0], sorted_x.get(), gate_out.get(),
                                  cumsum_i64.get(), n_hidden, n_ff, n_rows, n_exp);
        run_sorted_grouped_matmul(ctx, up_node->src[0], sorted_x.get(), up_out.get(),
                                  cumsum_i64.get(), n_hidden, n_ff, n_rows, n_exp);
    }

    int64_t half_ne[2] = { n_ff, n_rows };
    size_t  half_nb[2] = { f16, n_ff * f16 };
    size_t  wide_nb[2] = { f16, 2 * n_ff * f16 };
    acl_tensor_ptr acl_gate = ggml_cann_create_tensor(
        gate_out.get(), ACL_FLOAT16, f16, half_ne,
        merged_gate_up ? wide_nb : half_nb, 2);
    acl_tensor_ptr acl_up = ggml_cann_create_tensor(
        merged_gate_up ? (char *) gate_out.get() + n_ff * f16 : up_out.get(),
        ACL_FLOAT16, f16, half_ne,
        merged_gate_up ? wide_nb : half_nb, 2);
    acl_tensor_ptr acl_act = ggml_cann_create_tensor(
        activation.get(), ACL_FLOAT16, f16, half_ne, half_nb, 2);
    GGML_CANN_CALL_ACLNN_OP(ctx, Silu, acl_gate.get(), acl_act.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Mul, acl_act.get(), acl_up.get(), acl_act.get());

    run_sorted_grouped_matmul(ctx, down_node->src[0], activation.get(),
                              down_out.get(), cumsum_i64.get(), n_ff, n_embd,
                              n_rows, n_exp);

    int64_t probs_ne[2] = { n_used, n_token };
    size_t  probs16_nb[2] = { f16, n_used * f16 };
    size_t  probs32_nb[2] = { f32, n_used * f32 };
    int64_t down_ne[2] = { n_embd, n_rows };
    size_t  down_nb[2] = { f16, n_embd * f16 };
    int64_t routed_ne[2] = { n_embd, n_token };
    size_t  routed_nb[2] = { f16, n_embd * f16 };
    acl_tensor_ptr acl_probs16 = ggml_cann_create_tensor(
        probs16.get(), ACL_FLOAT16, f16, probs_ne, probs16_nb, 2);
    acl_tensor_ptr acl_probs32 = ggml_cann_create_tensor(
        probs_data, ACL_FLOAT, f32, probs_ne, probs32_nb, 2);
    acl_tensor_ptr acl_down = ggml_cann_create_tensor(
        down_out.get(), ACL_FLOAT16, f16, down_ne, down_nb, 2);
    acl_tensor_ptr acl_routed = ggml_cann_create_tensor(
        routed.get(), ACL_FLOAT16, f16, routed_ne, routed_nb, 2);
    int64_t restore_shape[2] = { n_token, n_embd };
    acl_int_array_ptr restore = ggml_cann_create_int_array(restore_shape, 2);
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_probs32.get(), ACL_FLOAT16,
                            acl_probs16.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, MoeTokenUnpermute, acl_down.get(),
                            acl_row_idx.get(), acl_probs16.get(), false,
                            restore.get(), acl_routed.get());

    size_t dst_nb[2] = { ggml_type_size(moe_out->type),
                         n_embd * ggml_type_size(moe_out->type) };
    acl_tensor_ptr acl_moe_out = ggml_cann_create_tensor(
        moe_out, ACL_FLOAT, f32, routed_ne, dst_nb, 2);
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_routed.get(), ACL_FLOAT,
                            acl_moe_out.get());

    *last_fused_node = final_idx;
    return true;
}
void run_moe_gmm(ggml_backend_cann_context & ctx, const ggml_tensor * weight,
                 const ggml_tensor * input, const ggml_tensor * ids,
                 ggml_cann_pool_alloc & input_alloc,
                 ggml_cann_pool_alloc & sorted_alloc,
                 ggml_cann_pool_alloc & row_idx_alloc,
                 ggml_cann_pool_alloc & out_alloc, bool gather_rows) {
    const int64_t n_out   = weight->ne[1];
    const int64_t n_in    = weight->ne[0];
    const int64_t n_exp   = weight->ne[2];
    const int64_t n_used  = ids->ne[0];
    const int64_t n_token = ids->ne[1];
    const int64_t n_rows  = n_used * n_token;
    const int64_t x_rows  = input->ne[1] == 1 ? n_token : n_rows;

    GGML_ASSERT(input_alloc.get() != nullptr);
    GGML_ASSERT(input_alloc.actual_size >= n_in * x_rows * sizeof(uint16_t));
    GGML_ASSERT(sorted_alloc.get() != nullptr);
    GGML_ASSERT(sorted_alloc.actual_size >= n_rows * n_in * sizeof(uint16_t));
    GGML_ASSERT(row_idx_alloc.get() != nullptr);
    GGML_ASSERT(row_idx_alloc.actual_size >= n_rows * sizeof(int32_t));
    GGML_ASSERT(out_alloc.get() != nullptr);
    GGML_ASSERT(out_alloc.actual_size >= n_rows * n_out * sizeof(uint16_t));
    {
        int64_t ne[2]    = { n_in, x_rows };
        size_t src_nb[2] = { input->nb[0], input->nb[1] };
        size_t dst_nb[2] = { sizeof(uint16_t), n_in * sizeof(uint16_t) };
        acl_tensor_ptr src = ggml_cann_create_tensor(input->data, ACL_FLOAT, sizeof(float),
                                                     ne, src_nb, 2);
        acl_tensor_ptr dst = ggml_cann_create_tensor(input_alloc.get(), ACL_FLOAT16,
                                                     sizeof(uint16_t), ne, dst_nb, 2);
        GGML_CANN_CALL_ACLNN_OP(ctx, Cast, src.get(), ACL_FLOAT16, dst.get());
    }

    ggml_cann_pool_alloc cumsum32_alloc(ctx.pool(), n_exp * sizeof(int32_t));
    ggml_cann_pool_alloc cumsum64_alloc(ctx.pool(), n_exp * sizeof(int64_t));

    int64_t row_ne[1]    = { n_rows };
    size_t  row_nb[1]    = { sizeof(int32_t) };
    int64_t cumsum_ne[1] = { n_exp };
    size_t  c32_nb[1]    = { sizeof(int32_t) };
    size_t  c64_nb[1]    = { sizeof(int64_t) };

    int64_t input_ne[2] = { n_in, x_rows };
    size_t  input_nb[2] = { sizeof(uint16_t), n_in * sizeof(uint16_t) };
    int64_t routed_ne[2] = { n_in, n_rows };
    size_t  routed_nb[2] = { sizeof(uint16_t), n_in * sizeof(uint16_t) };
    acl_tensor_ptr acl_x = ggml_cann_create_tensor(
        input_alloc.get(), ACL_FLOAT16, sizeof(uint16_t), input_ne,
        input_nb, 2);
    acl_tensor_ptr acl_sorted_x = ggml_cann_create_tensor(
        sorted_alloc.get(), ACL_FLOAT16, sizeof(uint16_t), routed_ne, routed_nb, 2);
    acl_tensor_ptr acl_row_idx = ggml_cann_create_tensor(
        row_idx_alloc.get(), ACL_INT32, sizeof(int32_t), row_ne, row_nb, 1);
    acl_tensor_ptr acl_c32 = ggml_cann_create_tensor(
        cumsum32_alloc.get(), ACL_INT32, sizeof(int32_t), cumsum_ne, c32_nb, 1);
    acl_tensor_ptr acl_c64 = ggml_cann_create_tensor(
        cumsum64_alloc.get(), ACL_INT64, sizeof(int64_t), cumsum_ne, c64_nb, 1);

    acl_tensor_ptr acl_ids;
    auto * ids_mutable = const_cast<ggml_tensor *>(ids);
    size_t  ids_view_nb[2] = { ids->nb[0], ids->nb[1] };
    ggml_cann_pool_alloc ids_cont_alloc;
    ggml_cann_pool_alloc ids_expanded_alloc;
    ggml_cann_pool_alloc ids_perm_alloc;
    void * ids_data = ids->data;
    size_t ids_nb[2] = { ids->nb[0], ids->nb[1] };
    if (!ggml_is_contiguous(ids)) {
        ids_cont_alloc.alloc(ctx.pool(), n_rows * sizeof(int32_t));
        ids_data = ids_cont_alloc.get();
        ids_nb[0] = sizeof(int32_t);
        ids_nb[1] = n_used * sizeof(int32_t);
        int64_t ids_ne[2] = { n_used, n_token };
        acl_tensor_ptr src = ggml_cann_create_tensor(ids_mutable, ids_mutable->ne,
                                                     ids_view_nb, 2);
        acl_tensor_ptr dst = ggml_cann_create_tensor(ids_data, ACL_INT32, sizeof(int32_t),
                                                     ids_ne, ids_nb, 2);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, dst.get(), src.get());
    }
    if (x_rows == n_token) {
        int64_t ids_ne[2] = { n_used, n_token };
        acl_ids = ggml_cann_create_tensor(ids_data, ACL_INT32, sizeof(int32_t),
                                          ids_ne, ids_nb, 2);
    } else {
        std::vector<int32_t> perm(n_rows);
        for (int64_t token = 0; token < n_token; ++token) {
            for (int64_t used = 0; used < n_used; ++used) {
                perm[token * n_used + used] = (int32_t) (used * n_token + token);
            }
        }
        ids_perm_alloc.alloc(ctx.pool(), n_rows * sizeof(int32_t));
        ids_expanded_alloc.alloc(ctx.pool(), n_rows * sizeof(int32_t));
        ACL_CHECK(aclrtMemcpy(ids_perm_alloc.get(), n_rows * sizeof(int32_t),
                              perm.data(), n_rows * sizeof(int32_t),
                              ACL_MEMCPY_HOST_TO_DEVICE));
        int64_t flat_ne[1] = { n_rows };
        size_t  flat_nb[1] = { sizeof(int32_t) };
        acl_tensor_ptr acl_ids_flat = ggml_cann_create_tensor(
            ids_data, ACL_INT32, sizeof(int32_t), flat_ne, flat_nb, 1);
        acl_tensor_ptr acl_perm = ggml_cann_create_tensor(
            ids_perm_alloc.get(), ACL_INT32, sizeof(int32_t), flat_ne, flat_nb, 1);
        acl_tensor_ptr acl_ids_exp = ggml_cann_create_tensor(
            ids_expanded_alloc.get(), ACL_INT32, sizeof(int32_t), flat_ne, flat_nb, 1);
        GGML_CANN_CALL_ACLNN_OP(ctx, GatherV2, acl_ids_flat.get(), 0,
                                acl_perm.get(), acl_ids_exp.get());
        int64_t ids_ne[2] = { 1, n_rows };
        size_t  ids_exp_nb[2] = { sizeof(int32_t), sizeof(int32_t) };
        acl_ids = ggml_cann_create_tensor(ids_expanded_alloc.get(), ACL_INT32,
                                          sizeof(int32_t), ids_ne, ids_exp_nb, 2);
    }

    GGML_CANN_CALL_ACLNN_OP(ctx, MoeInitRoutingV2, acl_x.get(), acl_ids.get(), n_rows,
                            0, n_exp, 0, 1, false, acl_sorted_x.get(), acl_row_idx.get(),
                            acl_c32.get(), nullptr);
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_c32.get(), ACL_INT64, acl_c64.get());

    int64_t weight_ne[3] = { n_out, n_in, n_exp };
    size_t  weight_nb[3] = { weight->nb[1], weight->nb[0], weight->nb[2] };
    const aclFormat weight_format = ggml_cann_is_weight_layout_candidate(weight)
                                        ? ACL_FORMAT_FRACTAL_NZ : ACL_FORMAT_ND;
    acl_tensor_ptr acl_weight = ggml_cann_create_tensor(
        weight->data, ACL_FLOAT16, sizeof(uint16_t), weight_ne, weight_nb, 3,
        weight_format);

    int64_t gmm_out_ne[2] = { n_out, n_rows };
    size_t  gmm_out_nb[2] = { sizeof(uint16_t), n_out * sizeof(uint16_t) };
    acl_tensor_ptr acl_gmm_out = ggml_cann_create_tensor(
        out_alloc.get(), ACL_FLOAT16, sizeof(uint16_t), gmm_out_ne, gmm_out_nb, 2);

    acl_tensor_list_ptr x_list = ggml_cann_create_tensor_list(std::move(acl_sorted_x));
    acl_tensor_list_ptr w_list = ggml_cann_create_tensor_list(std::move(acl_weight));
    acl_tensor_list_ptr out_list = ggml_cann_create_tensor_list(std::move(acl_gmm_out));
    GGML_CANN_CALL_ACLNN_OP(ctx, GroupedMatmulV4, x_list.get(), w_list.get(), nullptr,
                            nullptr, nullptr, nullptr, nullptr, nullptr, acl_c64.get(),
                            nullptr, nullptr, nullptr, 2, 0, 0, 0, out_list.get(),
                            nullptr, nullptr);

    if (!gather_rows) {
        return;
    }

    std::vector<int32_t> host_rows(n_rows);
    std::vector<int32_t> host_ids(n_rows);
    std::vector<int32_t> host_cumsum(n_exp);
    std::vector<int32_t> inverse(n_rows, -1);
    ACL_CHECK(aclrtMemcpy(host_rows.data(), n_rows * sizeof(int32_t),
                          row_idx_alloc.get(), n_rows * sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(host_ids.data(), n_rows * sizeof(int32_t), ids_data,
                          n_rows * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(host_cumsum.data(), n_exp * sizeof(int32_t),
                          cumsum32_alloc.get(), n_exp * sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    for (int64_t sorted_row = 0; sorted_row < n_rows; ++sorted_row) {
        const int64_t token = host_rows[sorted_row];
        if (token < 0 || token >= n_token) {
            continue;
        }
        const int64_t expert = std::upper_bound(host_cumsum.begin(), host_cumsum.end(),
                                                (int32_t) sorted_row) - host_cumsum.begin();
        for (int64_t slot = 0; slot < n_used; ++slot) {
            if (host_ids[token * n_used + slot] == (int32_t) expert) {
                inverse[token * n_used + slot] = (int32_t) sorted_row;
                break;
            }
        }
    }
    ggml_cann_pool_alloc inverse_alloc(ctx.pool(), n_rows * sizeof(int32_t));
    ggml_cann_pool_alloc gathered_alloc(ctx.pool(), n_rows * n_out * sizeof(uint16_t));
    ACL_CHECK(aclrtMemcpy(inverse_alloc.get(), n_rows * sizeof(int32_t),
                          inverse.data(), n_rows * sizeof(int32_t),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    acl_tensor_ptr acl_sorted = ggml_cann_create_tensor(
        out_alloc.get(), ACL_FLOAT16, sizeof(uint16_t), gmm_out_ne, gmm_out_nb, 2);
    acl_tensor_ptr acl_gathered = ggml_cann_create_tensor(
        gathered_alloc.get(), ACL_FLOAT16, sizeof(uint16_t), gmm_out_ne, gmm_out_nb, 2);
    int64_t inverse_ne[1] = { n_rows };
    size_t  inverse_nb[1] = { sizeof(int32_t) };
    acl_tensor_ptr acl_inverse = ggml_cann_create_tensor(
        inverse_alloc.get(), ACL_INT32, sizeof(int32_t), inverse_ne, inverse_nb, 1);
    GGML_CANN_CALL_ACLNN_OP(ctx, GatherV2, acl_sorted.get(), 0, acl_inverse.get(),
                            acl_gathered.get());
    ACL_CHECK(aclrtMemcpyAsync(out_alloc.get(), n_rows * n_out * sizeof(uint16_t),
                               gathered_alloc.get(), n_rows * n_out * sizeof(uint16_t),
                               ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()));
}

} // namespace

bool ggml_cann_moe_router_fused(ggml_backend_cann_context & ctx,
                                const ggml_cgraph * graph, int node_idx,
                                int * last_fused_node) {
    static const bool enabled = env_enabled("GGML_CANN_MOE_ROUTER_FUSION", true);
    static const bool debug =
        env_enabled("GGML_CANN_MOE_FULL_DEBUG", false);
    static int debug_count = 0;
    ggml_tensor * node = graph->nodes[node_idx];
    if (!enabled || node->op != GGML_OP_MUL_MAT || node->src[0] == nullptr ||
            node->src[1] == nullptr || !is_router_weight(node->src[0])) {
        return false;
    }

    ggml_tensor * selected = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_tensor * probs = find_router_nodes(graph, node_idx, &selected, &weights);
    if (debug && debug_count++ < 12) {
        fprintf(stderr,
                "[MoE ROUTER] idx=%d logits=%s ne=%ldx%ld cur=%s ne=%ldx%ld "
                "probs=%d sel=%d w=%d\n",
                node_idx, tensor_name(node->src[0]), (long) node->src[0]->ne[0],
                (long) node->src[0]->ne[1], tensor_name(node->src[1]),
                (long) node->src[1]->ne[0], (long) node->src[1]->ne[1],
                find_node_index(graph, probs), find_node_index(graph, selected),
                find_node_index(graph, weights));
    }
    if (probs == nullptr || selected == nullptr || weights == nullptr) {
        return false;
    }

    const int last = find_node_index(graph, weights);
    if (last < 0) {
        return false;
    }

    const int64_t n_token = node->src[1]->ne[1];
    const int64_t n_exp   = node->src[0]->ne[1];
    const int64_t n_used  = selected->ne[0];
    if (n_exp % 32 != 0) {
        return false;
    }
    const bool v2_routing = false;
    constexpr int64_t chunk_rows = 3072;
    const int64_t chunks = (n_token + chunk_rows - 1) / chunk_rows;

    ggml_cann_pool_alloc logits16_alloc(ctx.pool(), n_token * n_exp * sizeof(uint16_t));
    {
        int64_t logits_ne[2] = { n_exp, n_token };
        size_t  src_nb[2]    = { sizeof(float), n_exp * sizeof(float) };
        size_t  dst_nb[2]    = { sizeof(uint16_t), n_exp * sizeof(uint16_t) };
        acl_tensor_ptr src = ggml_cann_create_tensor(node->data, ACL_FLOAT,
                                                     sizeof(float), logits_ne, src_nb, 2);
        acl_tensor_ptr dst = ggml_cann_create_tensor(logits16_alloc.get(), ACL_FLOAT16,
                                                     sizeof(uint16_t), logits_ne, dst_nb, 2);
        GGML_CANN_CALL_ACLNN_OP(ctx, Cast, src.get(), ACL_FLOAT16, dst.get());
    }
    ggml_cann_pool_alloc topk_w16(ctx.pool(), n_token * n_used * sizeof(uint16_t));
    ggml_cann_pool_alloc topk_ids(ctx.pool(), n_token * n_used * sizeof(int32_t));
    ggml_cann_pool_alloc row_idx(ctx.pool(), n_token * n_used * sizeof(int32_t));
    ggml_cann_pool_alloc topk_w32(ctx.pool(), n_token * n_used * sizeof(float));
    uint64_t total_ws = 0;
    std::vector<uint64_t> ws_sizes;
    std::vector<aclOpExecutor *> executors;
    if (v2_routing) {
        int64_t logits_ne[2] = { n_exp, n_token };
        size_t  logits_nb[2] = { sizeof(uint16_t), n_exp * sizeof(uint16_t) };
        int64_t out_ne[2]    = { n_used, n_token };
        size_t  out32_nb[2]  = { sizeof(float), n_used * sizeof(float) };
        size_t  ids_nb[2]    = { sizeof(int32_t), n_used * sizeof(int32_t) };
        acl_tensor_ptr acl_logits = ggml_cann_create_tensor(
            logits16_alloc.get(), ACL_FLOAT16, sizeof(uint16_t),
            logits_ne, logits_nb, 2);
        acl_tensor_ptr acl_y32 = ggml_cann_create_tensor(
            topk_w32.get(), ACL_FLOAT, sizeof(float), out_ne, out32_nb, 2);
        acl_tensor_ptr acl_ids = ggml_cann_create_tensor(
            topk_ids.get(), ACL_INT32, sizeof(int32_t), out_ne, ids_nb, 2);
        GGML_CANN_CALL_ACLNN_OP(ctx, MoeGatingTopKSoftmaxV2, acl_logits.get(),
                                nullptr, n_used, 0, false, acl_y32.get(),
                                acl_ids.get(), nullptr);
    } else {
        for (int64_t c = 0; c < chunks; ++c) {
            const int64_t offset = c * chunk_rows;
            const int64_t rows   = std::min(chunk_rows, n_token - offset);
            int64_t logits_ne[2] = { n_exp, rows };
            size_t  logits_nb[2] = { sizeof(uint16_t), n_exp * sizeof(uint16_t) };
            int64_t out_ne[2]    = { n_used, rows };
            size_t  out_nb[2]    = { sizeof(uint16_t), n_used * sizeof(uint16_t) };
            size_t  ids_nb[2]    = { sizeof(int32_t), n_used * sizeof(int32_t) };
            size_t  logits_off   = offset * n_exp * sizeof(uint16_t);
            size_t  out_off_w    = offset * n_used * sizeof(uint16_t);
            size_t  out_off_i    = offset * n_used * sizeof(int32_t);
            auto logits = ggml_cann_create_tensor((char *) logits16_alloc.get() + logits_off,
                                                  ACL_FLOAT16, sizeof(uint16_t),
                                                  logits_ne, logits_nb, 2);
            auto w_out = ggml_cann_create_tensor((char *) topk_w16.get() + out_off_w, ACL_FLOAT16,
                                                 sizeof(uint16_t), out_ne, out_nb, 2);
            auto id_out = ggml_cann_create_tensor((char *) topk_ids.get() + out_off_i, ACL_INT32,
                                                  sizeof(int32_t), out_ne, ids_nb, 2);
            auto rid_out = ggml_cann_create_tensor((char *) row_idx.get() + out_off_i, ACL_INT32,
                                                   sizeof(int32_t), out_ne, ids_nb, 2);
            uint64_t ws = 0;
            aclOpExecutor * executor = nullptr;
            ACL_CHECK(aclnnMoeGatingTopKSoftmaxGetWorkspaceSize(
                logits.get(), nullptr, n_used, w_out.get(), id_out.get(), rid_out.get(),
                &ws, &executor));
            total_ws += ws;
            ws_sizes.push_back(ws);
            executors.push_back(executor);
        }

        ggml_cann_pool_alloc ws_alloc(ctx.pool(), total_ws);
        char * ws_ptr = (char *) ws_alloc.get();
        for (size_t c = 0; c < executors.size(); ++c) {
            ACL_CHECK(aclnnMoeGatingTopKSoftmax(ws_ptr, ws_sizes[c], executors[c], ctx.stream()));
            ws_ptr += ws_sizes[c];
        }
    }

    int64_t topk_ne[2] = { n_used, n_token };
    size_t  topk_nb[2] = { sizeof(uint16_t), n_used * sizeof(uint16_t) };
    size_t  topk32_nb[2] = { sizeof(float), n_used * sizeof(float) };
    ggml_cann_pool_alloc sum32(ctx.pool(), n_token * sizeof(float));
    ggml_cann_pool_alloc norm32(ctx.pool(), n_token * n_used * sizeof(float));

    acl_tensor_ptr acl_w16 = ggml_cann_create_tensor(topk_w16.get(), ACL_FLOAT16,
                                                     sizeof(uint16_t), topk_ne, topk_nb, 2);
    acl_tensor_ptr acl_w32 = ggml_cann_create_tensor(topk_w32.get(), ACL_FLOAT,
                                                     sizeof(float), topk_ne, topk32_nb, 2);
    if (!v2_routing) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_w16.get(), ACL_FLOAT, acl_w32.get());
    }

    int64_t sum_ne[2] = { 1, n_token };
    size_t  sum_nb[2] = { sizeof(float), sizeof(float) };
    acl_tensor_ptr acl_sum = ggml_cann_create_tensor(sum32.get(), ACL_FLOAT,
                                                     sizeof(float), sum_ne, sum_nb, 2);
    int64_t reduce_dims_data[1] = { 1 };
    acl_int_array_ptr reduce_dims = ggml_cann_create_int_array(reduce_dims_data, 1);
    GGML_CANN_CALL_ACLNN_OP(ctx, ReduceSum, acl_w32.get(), reduce_dims.get(), true,
                            ACL_FLOAT, acl_sum.get());
    acl_tensor_ptr acl_norm32 = ggml_cann_create_tensor(norm32.get(), ACL_FLOAT,
                                                        sizeof(float), topk_ne, topk32_nb, 2);
    GGML_CANN_CALL_ACLNN_OP(ctx, Div, acl_w32.get(), acl_sum.get(), acl_norm32.get());

    static const bool full_fusion =
        env_enabled("GGML_CANN_MOE_FULL_FUSION", true);
    if (full_fusion &&
            run_moe_full_fused(ctx, graph, node_idx, last, selected, weights,
                               &topk_ids, &norm32, last_fused_node)) {
        return true;
    }

    for (int64_t token = 0; token < n_token; ++token) {
        ACL_CHECK(aclrtMemcpyAsync(
            (char *) selected->data + token * selected->nb[1], n_used * sizeof(int32_t),
            (char *) topk_ids.get() + token * n_used * sizeof(int32_t),
            n_used * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()));
        ACL_CHECK(aclrtMemcpyAsync(
            (char *) weights->data + token * weights->nb[2], n_used * sizeof(float),
            (char *) norm32.get() + token * n_used * sizeof(float),
            n_used * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()));
    }

    *last_fused_node = last;
    return true;
}

bool ggml_cann_moe_gate_up_fused(ggml_backend_cann_context & ctx,
                                 const ggml_cgraph * graph, int node_idx,
                                 int * last_fused_node) {
    static const bool enabled = env_enabled("GGML_CANN_MOE_SWIGLU_FUSION", false);
    static const bool full_fusion =
        env_enabled("GGML_CANN_MOE_FULL_FUSION", true);
    static const bool debug =
        env_enabled("GGML_CANN_MOE_FULL_DEBUG", false);
    static int debug_count = 0;
    ggml_tensor * node = graph->nodes[node_idx];
    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] == nullptr) {
        return false;
    }

    if (full_fusion && (is_gate_up_weight(node->src[0]) ||
                            is_gate_weight(node->src[0]) ||
                            is_up_weight(node->src[0])) &&
            ggml_cann_mul_mat_id_grouped_supported(node)) {
        ggml_tensor * selected = node->src[2];
        const ggml_tensor * logits =
            selected != nullptr && selected->src[0] != nullptr &&
            selected->src[0]->src[0] != nullptr &&
            selected->src[0]->src[0]->src[0] != nullptr
                ? selected->src[0]->src[0]->src[0]
                : nullptr;
        const int router_idx = find_node_index(graph, logits);
        const int selected_idx = find_node_index(graph, selected);
        if (debug && debug_count++ < 12) {
            fprintf(stderr,
                    "[MoE FULL GATE] idx=%d weight=%s router=%d sel=%d "
                    "logits=%d\n",
                    node_idx, tensor_name(node->src[0]), router_idx,
                    selected_idx, find_node_index(graph, logits));
        }
        if (router_idx >= 0 && selected_idx > router_idx) {
            ggml_tensor * router_selected = nullptr;
            ggml_tensor * router_weights = nullptr;
            find_router_nodes(graph, router_idx, &router_selected,
                              &router_weights);
            if (debug && router_selected != selected) {
                fprintf(stderr, "[MoE FULL GATE] selected mismatch\n");
            } else if (debug && router_weights == nullptr) {
                fprintf(stderr, "[MoE FULL GATE] no weights\n");
            }
            if (router_selected == selected && router_weights != nullptr &&
                    run_moe_full_fused(ctx, graph, router_idx, selected_idx,
                                       selected, router_weights, nullptr,
                                       nullptr, last_fused_node)) {
                return true;
            }
        }
    }
    if (!enabled) {
        return false;
    }

    ggml_tensor * gate_node = nullptr;
    ggml_tensor * up_node = nullptr;
    if (is_gate_up_weight(node->src[0]) &&
            ggml_cann_mul_mat_id_grouped_supported(node)) {
        gate_node = node;
        up_node = node;
    } else if (is_gate_weight(node->src[0])) {
        gate_node = node;
        for (int i = node_idx + 1; i < graph->n_nodes; ++i) {
            ggml_tensor * candidate = graph->nodes[i];
            if (candidate->op == GGML_OP_MUL_MAT_ID && candidate->src[0] != nullptr &&
                    is_up_weight(candidate->src[0]) && candidate->src[2] == node->src[2]) {
                up_node = candidate;
                break;
            }
        }
    } else if (is_up_weight(node->src[0])) {
        up_node = node;
        for (int i = node_idx + 1; i < graph->n_nodes; ++i) {
            ggml_tensor * candidate = graph->nodes[i];
            if (candidate->op == GGML_OP_MUL_MAT_ID && candidate->src[0] != nullptr &&
                    is_gate_weight(candidate->src[0]) && candidate->src[2] == node->src[2]) {
                gate_node = candidate;
                break;
            }
        }
    }
    if (gate_node == nullptr || up_node == nullptr) {
        return false;
    }
    if (!ggml_cann_mul_mat_id_grouped_supported(gate_node) ||
            !ggml_cann_mul_mat_id_grouped_supported(up_node)) {
        return false;
    }

    ggml_tensor * glu = gate_node == up_node
        ? find_glu_node(graph, node_idx)
        : find_glu_node_for_pair(graph, node_idx, gate_node, up_node);
    if (glu == nullptr || ggml_get_op_params_i32(glu, 1) != 0) {
        return false;
    }
    const int glu_idx = find_node_index(graph, glu);
    if (glu_idx < 0) {
        return false;
    }

    const int64_t n_ff = node->ne[0] / 2;
    const int64_t n_rows = node->ne[1] * node->ne[2];
    const int64_t n_token = node->ne[2];
    const int64_t n_hidden = gate_node->src[0]->ne[0];
    ggml_cann_pool_alloc hidden16(ctx.pool(), n_hidden * n_token * sizeof(uint16_t));
    ggml_cann_pool_alloc gate_out(ctx.pool(),
                                  ggml_nelements(gate_node) * sizeof(uint16_t));
    ggml_cann_pool_alloc gate_sorted(ctx.pool(), n_rows * n_hidden * sizeof(uint16_t));
    ggml_cann_pool_alloc gate_rows(ctx.pool(), n_rows * sizeof(int32_t));
    ggml_cann_pool_alloc up_out;
    ggml_cann_pool_alloc up_sorted;
    ggml_cann_pool_alloc up_rows;
    if (gate_node == up_node) {
        run_moe_gmm(ctx, gate_node->src[0], gate_node->src[1], gate_node->src[2],
                    hidden16, gate_sorted, gate_rows, gate_out, true);
    } else {
        up_out.alloc(ctx.pool(), ggml_nelements(up_node) * sizeof(uint16_t));
        up_sorted.alloc(ctx.pool(), n_rows * n_hidden * sizeof(uint16_t));
        up_rows.alloc(ctx.pool(), n_rows * sizeof(int32_t));
        run_moe_gmm(ctx, gate_node->src[0], gate_node->src[1], gate_node->src[2],
                    hidden16, gate_sorted, gate_rows, gate_out, true);
        run_moe_gmm(ctx, up_node->src[0], up_node->src[1], up_node->src[2],
                    hidden16, up_sorted, up_rows, up_out, true);
    }

    ggml_cann_pool_alloc act_alloc(ctx.pool(), n_rows * n_ff * sizeof(uint16_t));
    int64_t half_ne[2] = { n_ff, n_rows };
    size_t  half_nb[2] = { sizeof(uint16_t), n_ff * sizeof(uint16_t) };
    size_t  wide_nb[2] = { sizeof(uint16_t), (2 * n_ff) * sizeof(uint16_t) };
    acl_tensor_ptr gate = ggml_cann_create_tensor(gate_out.get(), ACL_FLOAT16,
                                                  sizeof(uint16_t), half_ne, wide_nb, 2);
    acl_tensor_ptr up = ggml_cann_create_tensor(
        (gate_node == up_node ? (char *) gate_out.get() + n_ff * sizeof(uint16_t)
                              : (char *) up_out.get()),
        ACL_FLOAT16, sizeof(uint16_t), half_ne, wide_nb, 2);
    acl_tensor_ptr act = ggml_cann_create_tensor(act_alloc.get(), ACL_FLOAT16,
                                                 sizeof(uint16_t), half_ne, half_nb, 2);
    GGML_CANN_CALL_ACLNN_OP(ctx, Silu, gate.get(), act.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Mul, act.get(), up.get(), act.get());

    int64_t dst_ne[2] = { n_ff, n_rows };
    size_t  dst_nb[2] = { ggml_type_size(glu->type), n_ff * ggml_type_size(glu->type) };
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(glu, ACL_FLOAT,
                                                     sizeof(float), dst_ne, dst_nb, 2);
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, act.get(), ACL_FLOAT, acl_dst.get());
    *last_fused_node = glu_idx;
    return true;
}

bool ggml_cann_moe_down_unpermute_fused(ggml_backend_cann_context & ctx,
                                        const ggml_cgraph * graph, int node_idx,
                                        int * last_fused_node) {
    static const bool enabled = env_enabled("GGML_CANN_MOE_UNPERMUTE_FUSION", true);
    ggml_tensor * node = graph->nodes[node_idx];
    if (!enabled || node->op != GGML_OP_MUL_MAT_ID || node->src[0] == nullptr ||
            !is_down_weight(node->src[0]) ||
            !ggml_cann_mul_mat_id_grouped_supported(node)) {
        return false;
    }

    ggml_tensor * weighted = nullptr;
    if (find_weighted_down(graph, node_idx, &weighted) == nullptr) {
        return false;
    }
    const int weighted_idx = find_node_index(graph, weighted);
    ggml_tensor * moe_out = find_moe_output(graph, weighted_idx, node, weighted);
    if (moe_out == nullptr || moe_out->type != GGML_TYPE_F32 ||
            moe_out->ne[0] != node->src[0]->ne[0] ||
            moe_out->ne[1] != node->ne[2]) {
        return false;
    }
    const int final_idx = find_node_index(graph, moe_out);
    if (final_idx < weighted_idx) {
        return false;
    }

    const int64_t n_token = node->ne[2];
    const int64_t n_used  = node->ne[1];
    const int64_t n_embd  = node->ne[0];
    const int64_t n_ff = node->src[0]->ne[1];
    ggml_cann_pool_alloc input16(ctx.pool(), n_ff * n_used * n_token * sizeof(uint16_t));
    ggml_cann_pool_alloc gmm_out_alloc(ctx.pool(),
                                       ggml_nelements(node) * sizeof(uint16_t));
    ggml_cann_pool_alloc sorted_alloc(ctx.pool(),
                                      n_used * n_token * n_ff * sizeof(uint16_t));
    ggml_cann_pool_alloc row_idx_alloc(ctx.pool(), n_used * n_token * sizeof(int32_t));
    run_moe_gmm(ctx, node->src[0], node->src[1], node->src[2], input16, sorted_alloc,
                row_idx_alloc, gmm_out_alloc, false);

    ggml_cann_pool_alloc probs16(ctx.pool(), n_token * n_used * sizeof(uint16_t));
    int64_t probs_ne[2] = { n_used, n_token };
    size_t  probs_nb[2] = { sizeof(uint16_t), n_used * sizeof(uint16_t) };
    size_t  weights_nb[2] = { sizeof(float), weighted->src[1]->nb[0] };
    acl_tensor_ptr acl_prob_dst = ggml_cann_create_tensor(probs16.get(), ACL_FLOAT16,
                                                          sizeof(uint16_t), probs_ne,
                                                          probs_nb, 2);
    acl_tensor_ptr acl_prob_src = ggml_cann_create_tensor(weighted->src[1], ACL_FLOAT,
                                                          sizeof(float), probs_ne,
                                                          weights_nb, 2);
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_prob_src.get(), ACL_FLOAT16, acl_prob_dst.get());

    ggml_cann_pool_alloc routed16(ctx.pool(), n_token * n_embd * sizeof(uint16_t));
    int64_t routed_ne[2] = { n_embd, n_token };
    size_t  routed_nb[2] = { sizeof(uint16_t), n_embd * sizeof(uint16_t) };
    acl_tensor_ptr acl_routed = ggml_cann_create_tensor(routed16.get(), ACL_FLOAT16,
                                                        sizeof(uint16_t), routed_ne,
                                                        routed_nb, 2);
    int64_t row_ne[1] = { n_used * n_token };
    size_t  row_nb[1] = { sizeof(int32_t) };
    acl_tensor_ptr acl_rows = ggml_cann_create_tensor(row_idx_alloc.get(), ACL_INT32,
                                                      sizeof(int32_t), row_ne, row_nb, 1);
    int64_t gmm_ne[2] = { n_embd, n_used * n_token };
    size_t  gmm_nb[2] = { sizeof(uint16_t), n_embd * sizeof(uint16_t) };
    acl_tensor_ptr acl_gmm_out = ggml_cann_create_tensor(
        gmm_out_alloc.get(), ACL_FLOAT16, sizeof(uint16_t), gmm_ne, gmm_nb, 2);
    int64_t restore_shape[2] = { n_token, n_embd };
    acl_int_array_ptr restore = ggml_cann_create_int_array(restore_shape, 2);
    GGML_CANN_CALL_ACLNN_OP(ctx, MoeTokenUnpermute, acl_gmm_out.get(), acl_rows.get(),
                            acl_prob_dst.get(), false, restore.get(), acl_routed.get());

    size_t  dst_nb[2] = { ggml_type_size(moe_out->type), n_embd * ggml_type_size(moe_out->type) };
    acl_tensor_ptr acl_moe_out = ggml_cann_create_tensor(moe_out, ACL_FLOAT,
                                                         sizeof(float), routed_ne,
                                                         dst_nb, 2);
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_routed.get(), ACL_FLOAT, acl_moe_out.get());
    *last_fused_node = final_idx;
    return true;
}
