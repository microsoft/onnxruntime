// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define HEAD_ELEMS 128
#define HEAD_DIM_NAMESPACE H128
#define XQA_PAGED_CACHE_ELEM 1
#define XQA_PAGED_INPUT_FP16 0
#define XQA_PAGED_QUERY_T __nv_bfloat16
#define XQA_PAGED_FAMILY bf16_int8
#define XQA_PAGED_LAUNCH_FN LaunchXQAPagedInt8KernelBF16

#include "xqa_paged_loader_impl.cuh"
