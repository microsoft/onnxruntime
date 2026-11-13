// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_INT4_KV_CACHE)
#include <cuda_bf16.h>

#define HEAD_ELEMS 128
#define HEAD_DIM_NAMESPACE H128
#define XQA_PAGED_CACHE_ELEM 0
#define XQA_PAGED_INT4 1
#define XQA_PAGED_INPUT_FP16 0
#define XQA_PAGED_QUERY_T __nv_bfloat16
#define XQA_PAGED_FAMILY bf16_int4_h128
#define XQA_PAGED_LAUNCH_FN LaunchXQAPagedInt4KernelBF16
#define XQA_PAGED_GROUP6_ONLY 1

#include "xqa_paged_loader_impl.cuh"
#endif
