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
#include <aclnnop/aclnn_sort.h>
#include <aclnnop/aclnn_unique_consecutive.h>
#include <aclnnop/aclnn_cumsum.h>

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

static int64_t acl_tensor_first_dim(const aclTensor * tensor) {
    int64_t * dims = nullptr;
    uint64_t dim_count = 0;
    ACL_CHECK(aclGetViewShape(tensor, &dims, &dim_count));
    GGML_ASSERT(dim_count == 1);
    return dims[0];
}

// Gate the 310P grouped W8A8 path on type, shape, layout, and routing shape.
// This is the first place to look when a new MoE model does not take the A8 path.
bool ggml_cann_mul_mat_id_grouped_supported(const ggml_tensor * dst) {
    const ggml_tensor * weight = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    const ggml_tensor * ids    = dst->src[2];
    const bool down_expert = is_down_expert_weight(weight);

    if (weight->type == GGML_TYPE_W8A8) {
        // The grouped A8 route requires the 3D W8A8 expert layout and an I32
        // routing tensor whose expanded shape matches the gate/up or down input.
        const bool supported = ggml_cann_w8a8_moe_a8_enabled() &&
               ggml_cann_w8a8_is_expert(weight) &&
               weight->ne[0] % 16 == 0 && weight->ne[1] % 16 == 0 &&
               weight->ne[3] == 1 &&
               (input->type == GGML_TYPE_F16 || input->type == GGML_TYPE_F32) &&
               input->ne[3] == 1 && ids->type == GGML_TYPE_I32 &&
               ((input->ne[1] == 1 && input->ne[2] == ids->ne[1]) ||
                (down_expert && input->ne[1] == ids->ne[0] && input->ne[2] == ids->ne[1])) &&
               ggml_is_contiguous(weight) && ggml_is_contiguous(input) &&
               ids->nb[0] == sizeof(int32_t);
        if (std::getenv("GGML_CANN_MOE_DEBUG") != nullptr) {
            fprintf(stderr,
                    "MoE support: dst=%s weight=%s a8=%d expert=%d w=[%lld,%lld,%lld,%lld] "
                    "input=[%lld,%lld,%lld,%lld] type=%s ids=[%lld,%lld] ids_type=%s nb0=%zu "
                    "weight_contig=%d input_contig=%d supported=%d\n",
                    ggml_get_name(dst), ggml_get_name(weight), ggml_cann_w8a8_moe_a8_enabled(),
                    ggml_cann_w8a8_is_expert(weight), (long long) weight->ne[0], (long long) weight->ne[1],
                    (long long) weight->ne[2], (long long) weight->ne[3], (long long) input->ne[0],
                    (long long) input->ne[1], (long long) input->ne[2], (long long) input->ne[3],
                    ggml_type_name(input->type), (long long) ids->ne[0], (long long) ids->ne[1],
                    ggml_type_name(ids->type), ids->nb[0], ggml_is_contiguous(weight),
                    ggml_is_contiguous(input), supported);
        }
        return supported;
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

    // ND is the stable path; NZ is experimental and default-off.
    const bool use_nz =
        parse_bool(get_env_as_lowercase("GGML_CANN_MOE_WEIGHT_NZ").value_or("off"));
    // CANN 9.1.0 can return incorrect results when most expert groups are empty.
    // Compact only the ND path, because the NZ operator consumes all experts.
    const bool compact_groups = weight->type == GGML_TYPE_W8A8 && !use_nz && n_rows < n_exp;

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
    acl_tensor_ptr acl_gmm_out = ggml_cann_create_tensor(gmm_out_alloc.get(), ACL_FLOAT16,
                                                         sizeof(uint16_t), gmm_out_ne, gmm_out_nb, 2);
    const aclFormat weight_format = ggml_cann_is_weight_layout_candidate(weight)
                                        ? ACL_FORMAT_FRACTAL_NZ : ACL_FORMAT_ND;
    acl_tensor_ptr acl_gmm_weight = ggml_cann_create_tensor(weight->data, ACL_FLOAT16,
                                                            sizeof(uint16_t), gmm_w_ne, gmm_w_nb, 3,
                                                            weight_format);

    if (weight->type == GGML_TYPE_W8A8) {
        char quant_mode[] = "pertoken";
        void * weight_scale_data = (char *) weight->data + ggml_cann_w8a8_expert_scale_offset(weight);

        if (compact_groups) {
            // Build a compact group list from only the experts that actually occur.
            // This avoids the CANN empty-group bug and keeps the operator input dense.
            ggml_cann_pool_alloc sorted_ids_alloc(ctx.pool(), n_rows * sizeof(int32_t));
            ggml_cann_pool_alloc sort_index_alloc(ctx.pool(), n_rows * sizeof(int64_t));
            ggml_cann_pool_alloc unique_ids_alloc(ctx.pool(), n_rows * sizeof(int32_t));
            ggml_cann_pool_alloc unique_counts_alloc(ctx.pool(), n_rows * sizeof(int64_t));
            ggml_cann_pool_alloc unique_inverse_alloc(ctx.pool(), n_rows * sizeof(int64_t));

            ACL_CHECK(aclrtMemsetAsync(unique_ids_alloc.get(), n_rows * sizeof(int32_t), 0,
                                       n_rows * sizeof(int32_t), ctx.stream()));
            ACL_CHECK(aclrtMemsetAsync(unique_counts_alloc.get(), n_rows * sizeof(int64_t), 0,
                                       n_rows * sizeof(int64_t), ctx.stream()));

            int64_t flat_ne[1] = { n_rows };
            int64_t ids_nb[1] = { 1 };
            int64_t i64_nb[1] = { 1 };
            acl_tensor_ptr acl_ids_flat = make_direct_tensor(routing_ids_data, flat_ne, ids_nb, 1,
                                                             ACL_INT32);
            acl_tensor_ptr acl_sorted_ids = make_direct_tensor(sorted_ids_alloc.get(), flat_ne,
                                                               ids_nb, 1, ACL_INT32);
            acl_tensor_ptr acl_sort_index = make_direct_tensor(sort_index_alloc.get(), flat_ne,
                                                               i64_nb, 1, ACL_INT64);
            acl_tensor_ptr acl_unique_ids = make_direct_tensor(unique_ids_alloc.get(), flat_ne,
                                                               ids_nb, 1, ACL_INT32);
            acl_tensor_ptr acl_unique_counts = make_direct_tensor(unique_counts_alloc.get(), flat_ne,
                                                                  i64_nb, 1, ACL_INT64);
            acl_tensor_ptr acl_unique_inverse = make_direct_tensor(unique_inverse_alloc.get(), flat_ne,
                                                                   i64_nb, 1, ACL_INT64);

            GGML_CANN_CALL_ACLNN_OP(ctx, Sort, acl_ids_flat.get(), true, 0, false,
                                    acl_sorted_ids.get(), acl_sort_index.get());
            GGML_CANN_CALL_ACLNN_OP(ctx, UniqueConsecutive, acl_sorted_ids.get(), false, true, 0,
                                    acl_unique_ids.get(), acl_unique_inverse.get(),
                                    acl_unique_counts.get());

            const int64_t unique_count = acl_tensor_first_dim(acl_unique_counts.get());
            GGML_ASSERT(unique_count > 0 && unique_count <= n_rows);

            ggml_cann_pool_alloc compact_weight_alloc(
                ctx.pool(), static_cast<size_t>(unique_count * n_ff * n_embd));
            ggml_cann_pool_alloc compact_scale_alloc(
                ctx.pool(), static_cast<size_t>(unique_count * n_ff) * sizeof(float));
            ggml_cann_pool_alloc compact_group_alloc(ctx.pool(), unique_count * sizeof(int64_t));

            int64_t compact_ne[1] = { unique_count };
            int64_t weight_ne[3] = { n_exp, n_ff, n_embd };
            int64_t weight_nb[3] = { n_ff * n_embd, n_embd, 1 };
            int64_t compact_weight_ne[3] = { unique_count, n_ff, n_embd };
            int64_t compact_weight_nb[3] = { n_ff * n_embd, n_embd, 1 };
            int64_t scale_ne[2] = { n_exp, n_ff };
            int64_t scale_nb[2] = { n_ff, 1 };
            int64_t compact_scale_ne[2] = { unique_count, n_ff };
            int64_t compact_scale_nb[2] = { n_ff, 1 };

            acl_tensor_ptr acl_compact_ids = make_direct_tensor(unique_ids_alloc.get(), compact_ne,
                                                                ids_nb, 1, ACL_INT32);
            acl_tensor_ptr acl_source_weight = make_direct_tensor(weight->data, weight_ne, weight_nb,
                                                                 3, ACL_INT8);
            acl_tensor_ptr acl_compact_weight = make_direct_tensor(compact_weight_alloc.get(),
                                                                   compact_weight_ne,
                                                                   compact_weight_nb, 3, ACL_INT8);
            acl_tensor_ptr acl_source_scale = make_direct_tensor(weight_scale_data, scale_ne,
                                                                 scale_nb, 2, ACL_FLOAT);
            acl_tensor_ptr acl_compact_scale = make_direct_tensor(compact_scale_alloc.get(),
                                                                 compact_scale_ne, compact_scale_nb,
                                                                 2, ACL_FLOAT);
            acl_tensor_ptr acl_compact_group = make_direct_tensor(compact_group_alloc.get(),
                                                                  compact_ne, i64_nb, 1,
                                                                  ACL_INT64);

            GGML_CANN_CALL_ACLNN_OP(ctx, Cumsum, acl_unique_counts.get(), 0, ACL_INT64,
                                    acl_compact_group.get());
            GGML_CANN_CALL_ACLNN_OP(ctx, GatherV2, acl_source_weight.get(), 0,
                                    acl_compact_ids.get(), acl_compact_weight.get());
            GGML_CANN_CALL_ACLNN_OP(ctx, GatherV2, acl_source_scale.get(), 0,
                                    acl_compact_ids.get(), acl_compact_scale.get());
            GGML_CANN_CALL_ACLNN_OP(ctx, QuantGroupedMatmulDequant,
                                    acl_gmm_in.get(), acl_compact_weight.get(),
                                    acl_compact_scale.get(), acl_compact_group.get(), nullptr,
                                    nullptr, nullptr, nullptr, quant_mode, true,
                                    acl_gmm_out.get());
        } else if (use_nz) {
            int64_t scale_ne[2] = { n_exp, n_ff };
            int64_t scale_nb[2] = { n_ff, 1 };
            acl_tensor_ptr acl_weight_scale = make_direct_tensor(weight_scale_data, scale_ne,
                                                                 scale_nb, 2, ACL_FLOAT);
            GGML_ASSERT(weight->extra != nullptr);
            aclTensor * transformed_weight = (aclTensor *) weight->extra;
            GGML_CANN_CALL_ACLNN_OP(ctx, QuantGroupedMatmulDequantWeightNZ,
                                    acl_gmm_in.get(), transformed_weight, acl_weight_scale.get(),
                                    acl_cumsum_i64.get(), nullptr, nullptr, nullptr, nullptr,
                                    quant_mode, true, acl_gmm_out.get());
        } else {
            int64_t scale_ne[2] = { n_exp, n_ff };
            int64_t scale_nb[2] = { n_ff, 1 };
            int64_t direct_weight_ne[3] = { n_exp, n_ff, n_embd };
            int64_t direct_weight_nb[3] = { n_ff * n_embd, n_embd, 1 };
            acl_tensor_ptr acl_weight_scale = make_direct_tensor(weight_scale_data, scale_ne,
                                                                 scale_nb, 2, ACL_FLOAT);
            acl_tensor_ptr acl_quant_weight = make_direct_tensor(weight->data, direct_weight_ne,
                                                                direct_weight_nb, 3, ACL_INT8);
            GGML_CANN_CALL_ACLNN_OP(ctx, QuantGroupedMatmulDequant,
                                    acl_gmm_in.get(), acl_quant_weight.get(),
                                    acl_weight_scale.get(), acl_cumsum_i64.get(), nullptr, nullptr,
                                    nullptr, nullptr, quant_mode, true, acl_gmm_out.get());
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
