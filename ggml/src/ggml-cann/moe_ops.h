#ifndef CANN_MOE_OPS_H
#define CANN_MOE_OPS_H

#include "acl_tensor.h"
#include "common.h"

#include "weight_layout.h"

void ggml_cann_mul_mat_id_grouped(ggml_backend_cann_context & ctx, ggml_tensor * dst);
bool ggml_cann_mul_mat_id_grouped_supported(const ggml_tensor * dst);

#endif
