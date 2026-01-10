// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100)
#pragma warning(disable : 4244)
#pragma warning(disable : 4200)
#endif

#include "contrib_ops/cuda/moe/ft_moe/moe_gemm_kernels_template.h"

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace ort_fastertransformer {
template class MoeGemmRunner<__nv_bfloat16, uint8_t>;
}  // namespace ort_fastertransformer
