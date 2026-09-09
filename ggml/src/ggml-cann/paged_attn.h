#pragma once

#include "common.h"

struct ggml_tensor;

// Paged KV cache attention for Ascend 310P via ATB ops.
// KV pools are wrapped as FRACTAL_NZ and interpreted by ATB ops; the rest of
// llama.cpp never sees the layout.
void ggml_cann_paged_attn(ggml_backend_cann_context & ctx, ggml_tensor * dst);
