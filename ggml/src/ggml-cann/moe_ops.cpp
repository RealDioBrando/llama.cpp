#include "moe_ops.h"

#include "aclnn_ops.h"

#include <aclnnop/aclnn_grouped_matmul_v4.h>
#include <aclnnop/aclnn_gather_v2.h>
#include <aclnnop/aclnn_copy.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_moe_init_routing_v2.h>
#include <aclnnop/aclnn_moe_token_unpermute.h>
#include <aclnnop/aclnn_quant_grouped_matmul_dequant.h>
#include <aclnnop/aclnn_quant_grouped_matmul_dequant_weight_nz.h>

#include <vector>
#include <cstring>
static bool is_down_expert_weight(const ggml_tensor * weight) {
    const char * name = ggml_get_name(weight);
    return name != nullptr && strstr(name, "ffn_down_exps.weight") != nullptr;
}

static acl_tensor_ptr make_direct_tensor(void * data,
                                         const int64_t * shape,
                                         const int64_t * strides,
                                         int64_t dims,
                                         aclDataType data_type,
                                         aclFormat format = ACL_FORMAT_ND) {
    int64_t storage_shape = 1;
    for (int64_t i = 0; i < dims; ++i) {
        storage_shape *= shape[i];
    }
    aclTensor * tensor = aclCreateTensor(shape, static_cast<uint64_t>(dims), data_type, strides, 0, format,
                                         &storage_shape, 1, data);
    return acl_tensor_ptr(tensor);
}

bool ggml_cann_mul_mat_id_grouped_supported(const ggml_tensor * dst) {
    const ggml_tensor * weight = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    const ggml_tensor * ids    = dst->src[2];
    const bool down_expert = is_down_expert_weight(weight);

    if (weight->type == GGML_TYPE_W8A8) {
        return ggml_cann_w8a8_a8_enabled() &&
               ggml_cann_w8a8_is_expert(weight) &&
               weight->ne[0] % 16 == 0 && weight->ne[1] % 16 == 0 &&
               weight->ne[3] == 1 &&
               (input->type == GGML_TYPE_F16 || input->type == GGML_TYPE_F32) &&
               input->ne[3] == 1 && ids->type == GGML_TYPE_I32 &&
               ((input->ne[1] == 1 && input->ne[2] == ids->ne[1]) ||
                (down_expert && input->ne[1] == ids->ne[0] && input->ne[2] == ids->ne[1])) &&
               ggml_is_contiguous(weight) && ggml_is_contiguous(input) &&
               ids->nb[0] == sizeof(int32_t);
    }

    return weight->type == GGML_TYPE_F16 && weight->ne[3] == 1 &&
           input->type == GGML_TYPE_F32 && input->ne[3] == 1 &&
           ids->type == GGML_TYPE_I32 &&
           weight->ne[0] % 32 == 0 &&
           weight->ne[1] % 32 == 0 &&
           ((input->ne[1] == 1 && input->ne[2] == ids->ne[1]) ||
           (down_expert && input->ne[1] == ids->ne[0] && input->ne[2] == ids->ne[1])) &&
           ggml_is_contiguous(weight) && ggml_is_contiguous(input) &&
           ids->nb[0] == sizeof(int32_t);
}

void ggml_cann_mul_mat_id_grouped(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * weight = dst->src[0];
    ggml_tensor * input  = dst->src[1];
    ggml_tensor * ids    = dst->src[2];

    const int64_t n_embd  = weight->ne[0];
    const int64_t n_ff    = weight->ne[1];
    const int64_t n_exp   = weight->ne[2];
    const int64_t n_used  = ids->ne[0];
    const int64_t n_token = ids->ne[1];
    const int64_t n_rows  = n_used * n_token;
    const bool is_down    = input->ne[1] != 1 && is_down_expert_weight(weight);
    const int64_t x_rows  = is_down ? n_rows : n_token;

    GGML_ASSERT(ggml_cann_mul_mat_id_grouped_supported(dst));

    ggml_cann_pool_alloc input_f16_alloc;
    void * input_f16_data = input->data;
    if (input->type != GGML_TYPE_F16) {
        input_f16_alloc.alloc(ctx.pool(), static_cast<size_t>(n_embd * x_rows) * sizeof(uint16_t));
        input_f16_data = input_f16_alloc.get();
        int64_t x_ne[2]     = { n_embd, x_rows };
        size_t  x_f32_nb[2] = { sizeof(float), n_embd * sizeof(float) };
        size_t  x_f16_nb[2] = { sizeof(uint16_t), n_embd * sizeof(uint16_t) };
        acl_tensor_ptr acl_src = ggml_cann_create_tensor(input->data, ACL_FLOAT, sizeof(float),
                                                         x_ne, x_f32_nb, 2);
        acl_tensor_ptr acl_dst = ggml_cann_create_tensor(input_f16_data, ACL_FLOAT16,
                                                         sizeof(uint16_t), x_ne, x_f16_nb, 2);
        GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_src.get(), ACL_FLOAT16, acl_dst.get());
    }

    ggml_cann_pool_alloc sorted_x_alloc(ctx.pool(), n_rows * n_embd * sizeof(uint16_t));
    ggml_cann_pool_alloc row_idx_alloc(ctx.pool(), n_rows * sizeof(int32_t));
    ggml_cann_pool_alloc cumsum_i32_alloc(ctx.pool(), n_exp * sizeof(int32_t));
    ggml_cann_pool_alloc cumsum_i64_alloc(ctx.pool(), n_exp * sizeof(int64_t));

    int64_t sorted_ne[2]     = { n_embd, n_rows };
    size_t  sorted_nb[2]     = { sizeof(uint16_t), n_embd * sizeof(uint16_t) };
    int64_t row_idx_ne[1]    = { n_rows };
    size_t  row_idx_nb[1]    = { sizeof(int32_t) };
    int64_t cumsum_ne[1]     = { n_exp };
    size_t  cumsum_i32_nb[1] = { sizeof(int32_t) };
    size_t  cumsum_i64_nb[1] = { sizeof(int64_t) };

    int64_t x_ne[2]     = { n_embd, x_rows };
    size_t  x_f16_nb[2] = { sizeof(uint16_t), n_embd * sizeof(uint16_t) };
    acl_tensor_ptr acl_x = ggml_cann_create_tensor(input_f16_data, ACL_FLOAT16,
                                                   sizeof(uint16_t), x_ne, x_f16_nb, 2);
    acl_tensor_ptr acl_routing_ids;
    ggml_cann_pool_alloc ids_cont_alloc;
    void * routing_ids_data = ids->data;
    size_t routing_ids_nb[2] = { ids->nb[0], ids->nb[1] };
    if (!ggml_is_contiguous(ids)) {
        // argsort_top_k returns a leading-k view with full-expert row stride.
        ids_cont_alloc.alloc(ctx.pool(), n_rows * sizeof(int32_t));
        routing_ids_data = ids_cont_alloc.get();
        routing_ids_nb[0] = sizeof(int32_t);
        routing_ids_nb[1] = n_used * sizeof(int32_t);

        int64_t ids_cont_ne[2] = { n_used, n_token };
        acl_tensor_ptr acl_ids_src = ggml_cann_create_tensor(ids, ids->ne, ids->nb, 2);
        acl_tensor_ptr acl_ids_dst = ggml_cann_create_tensor(
            routing_ids_data, ACL_INT32, sizeof(int32_t), ids_cont_ne, routing_ids_nb, 2);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_ids_dst.get(), acl_ids_src.get());
    }
    ggml_cann_pool_alloc perm_alloc;
    acl_tensor_ptr acl_expanded_ids;
    ggml_cann_pool_alloc expanded_ids_alloc;
    if (is_down) {
        // Flatten expert-major down input to rows. Expand token-major top-k ids
        // to the same row order before routing with one expert per row.
        std::vector<int32_t> perm(n_rows);
        for (int64_t u = 0; u < n_used; ++u) {
            for (int64_t t = 0; t < n_token; ++t) {
                perm[u * n_token + t] = (int32_t) (t * n_used + u);
            }
        }

        perm_alloc.alloc(ctx.pool(), n_rows * sizeof(int32_t));
        expanded_ids_alloc.alloc(ctx.pool(), n_rows * sizeof(int32_t));
        ACL_CHECK(aclrtMemcpy(perm_alloc.get(), n_rows * sizeof(int32_t), perm.data(),
                              n_rows * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE));

        int64_t flat_ne[1] = { n_rows };
        size_t  flat_nb[1] = { sizeof(int32_t) };
        acl_tensor_ptr acl_ids_flat = ggml_cann_create_tensor(routing_ids_data, ACL_INT32,
                                                              sizeof(int32_t), flat_ne, flat_nb, 1);
        acl_tensor_ptr acl_perm = ggml_cann_create_tensor(perm_alloc.get(), ACL_INT32,
                                                          sizeof(int32_t), flat_ne, flat_nb, 1);
        acl_tensor_ptr acl_ids_expanded = ggml_cann_create_tensor(expanded_ids_alloc.get(), ACL_INT32,
                                                                  sizeof(int32_t), flat_ne, flat_nb, 1);
        GGML_CANN_CALL_ACLNN_OP(ctx, GatherV2, acl_ids_flat.get(), 0, acl_perm.get(),
                                acl_ids_expanded.get());
        int64_t routing_ids_ne[2] = { 1, n_rows };
        size_t  routing_ids_nb[2] = { sizeof(int32_t), sizeof(int32_t) };
        acl_routing_ids = ggml_cann_create_tensor(expanded_ids_alloc.get(), ACL_INT32,
                                                  sizeof(int32_t), routing_ids_ne, routing_ids_nb, 2);
    } else {
        int64_t ids_ne[2] = { n_used, n_token };
        acl_routing_ids = ggml_cann_create_tensor(routing_ids_data, ggml_cann_type_mapping(ids->type),
                                                  ggml_type_size(ids->type), ids_ne, routing_ids_nb, 2);
    }
    acl_tensor_ptr acl_sorted_x = ggml_cann_create_tensor(sorted_x_alloc.get(), ACL_FLOAT16,
                                                          sizeof(uint16_t), sorted_ne, sorted_nb, 2);
    acl_tensor_ptr acl_row_idx = ggml_cann_create_tensor(row_idx_alloc.get(), ACL_INT32,
                                                         sizeof(int32_t), row_idx_ne, row_idx_nb, 1);
    acl_tensor_ptr acl_cumsum_i32 = ggml_cann_create_tensor(cumsum_i32_alloc.get(), ACL_INT32,
                                                            sizeof(int32_t), cumsum_ne, cumsum_i32_nb, 1);

    GGML_CANN_CALL_ACLNN_OP(ctx, MoeInitRoutingV2, acl_x.get(), acl_routing_ids.get(), n_rows, 0, n_exp,
                            0, 1, false, acl_sorted_x.get(), acl_row_idx.get(),
                            acl_cumsum_i32.get(), nullptr);

    acl_tensor_ptr acl_cumsum_i32_src = ggml_cann_create_tensor(cumsum_i32_alloc.get(), ACL_INT32,
                                                                sizeof(int32_t), cumsum_ne, cumsum_i32_nb, 1);
    acl_tensor_ptr acl_cumsum_i64 = ggml_cann_create_tensor(cumsum_i64_alloc.get(), ACL_INT64,
                                                            sizeof(int64_t), cumsum_ne, cumsum_i64_nb, 1);
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_cumsum_i32_src.get(), ACL_INT64, acl_cumsum_i64.get());

    int64_t gmm_in_ne[2]  = { n_embd, n_rows };
    size_t  gmm_in_nb[2]  = { sizeof(uint16_t), n_embd * sizeof(uint16_t) };
    int64_t gmm_w_ne[3]   = { n_ff, n_embd, n_exp };
    size_t  gmm_w_nb[3]   = { weight->nb[1], weight->nb[0], weight->nb[2] };
    int64_t gmm_out_ne[2] = { n_ff, n_rows };
    size_t  gmm_out_nb[2] = { sizeof(uint16_t), n_ff * sizeof(uint16_t) };

    ggml_cann_pool_alloc gmm_out_alloc(ctx.pool(), n_rows * n_ff * sizeof(uint16_t));
    acl_tensor_ptr acl_gmm_in = ggml_cann_create_tensor(sorted_x_alloc.get(), ACL_FLOAT16,
                                                        sizeof(uint16_t), gmm_in_ne, gmm_in_nb, 2);
    const aclFormat weight_format = ggml_cann_is_weight_layout_candidate(weight)
                                        ? ACL_FORMAT_FRACTAL_NZ : ACL_FORMAT_ND;
    acl_tensor_ptr acl_gmm_weight = ggml_cann_create_tensor(weight->data, ACL_FLOAT16,
                                                            sizeof(uint16_t), gmm_w_ne, gmm_w_nb, 3,
                                                            weight_format);
    acl_tensor_ptr acl_gmm_out = ggml_cann_create_tensor(gmm_out_alloc.get(), ACL_FLOAT16,
                                                         sizeof(uint16_t), gmm_out_ne, gmm_out_nb, 2);
    if (weight->type == GGML_TYPE_W8A8) {
        const bool use_nz =
            parse_bool(get_env_as_lowercase("GGML_CANN_MOE_WEIGHT_NZ").value_or("off"));
        int64_t direct_scale_ne[2] = { n_exp, n_ff };
        int64_t direct_scale_nb[2] = { n_ff, 1 };
        acl_tensor_ptr acl_weight_scale = make_direct_tensor(
            (char *) weight->data + ggml_cann_w8a8_expert_scale_offset(weight),
            direct_scale_ne, direct_scale_nb, 2, ACL_FLOAT);
        char quant_mode[] = "pertoken";
        if (use_nz) {
            GGML_ASSERT(weight->extra != nullptr);
            aclTensor * transformed_weight = (aclTensor *) weight->extra;
            GGML_CANN_CALL_ACLNN_OP(ctx, QuantGroupedMatmulDequantWeightNZ,
                                    acl_gmm_in.get(), transformed_weight, acl_weight_scale.get(),
                                    acl_cumsum_i64.get(), nullptr, nullptr, nullptr, nullptr,
                                    quant_mode, true, acl_gmm_out.get());
        } else {
            int64_t direct_weight_ne[3] = { n_exp, n_ff, n_embd };
            int64_t direct_weight_nb[3] = { n_ff * n_embd, n_embd, 1 };
            acl_tensor_ptr acl_quant_weight =
                make_direct_tensor(weight->data, direct_weight_ne, direct_weight_nb, 3, ACL_INT8);
            GGML_CANN_CALL_ACLNN_OP(ctx, QuantGroupedMatmulDequant,
                                    acl_gmm_in.get(), acl_quant_weight.get(), acl_weight_scale.get(),
                                    acl_cumsum_i64.get(), nullptr, nullptr, nullptr, nullptr,
                                    quant_mode, true, acl_gmm_out.get());
        }
    } else {
        acl_tensor_list_ptr acl_x_list = ggml_cann_create_tensor_list(std::move(acl_gmm_in));
        acl_tensor_list_ptr acl_w_list = ggml_cann_create_tensor_list(std::move(acl_gmm_weight));
        acl_tensor_list_ptr acl_out_list = ggml_cann_create_tensor_list(std::move(acl_gmm_out));

        GGML_CANN_CALL_ACLNN_OP(ctx, GroupedMatmulV4, acl_x_list.get(), acl_w_list.get(), nullptr,
                                nullptr, nullptr, nullptr, nullptr, nullptr, acl_cumsum_i64.get(),
                                nullptr, nullptr, nullptr, 2, 0, 0, 0, acl_out_list.get(),
                                nullptr, nullptr);
    }

    ggml_cann_pool_alloc out_f16_alloc(ctx.pool(), ggml_nelements(dst) * sizeof(uint16_t));
    int64_t out2d_ne[2] = { n_ff, n_rows };
    size_t  out2d_nb[2] = { sizeof(uint16_t), n_ff * sizeof(uint16_t) };
    acl_tensor_ptr acl_gmm_out_view = ggml_cann_create_tensor(gmm_out_alloc.get(), ACL_FLOAT16,
                                                              sizeof(uint16_t), gmm_out_ne, gmm_out_nb, 2);
    acl_tensor_ptr acl_out_f16 = ggml_cann_create_tensor(out_f16_alloc.get(), ACL_FLOAT16,
                                                         sizeof(uint16_t), out2d_ne, out2d_nb, 2);
    acl_tensor_ptr acl_row_idx_view = ggml_cann_create_tensor(row_idx_alloc.get(), ACL_INT32,
                                                              sizeof(int32_t), row_idx_ne, row_idx_nb, 1);
    GGML_CANN_CALL_ACLNN_OP(ctx, GatherV2, acl_gmm_out_view.get(), 0, acl_row_idx_view.get(),
                            acl_out_f16.get());

    size_t dst2d_nb[2] = { ggml_type_size(dst->type), n_ff * ggml_type_size(dst->type) };
    acl_tensor_ptr acl_dst2d = ggml_cann_create_tensor(dst->data, ggml_cann_type_mapping(dst->type),
                                                       ggml_type_size(dst->type), out2d_ne, dst2d_nb, 2);
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_out_f16.get(), ggml_cann_type_mapping(dst->type),
                            acl_dst2d.get());
}
