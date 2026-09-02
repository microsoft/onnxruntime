// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define HEAD_ELEMS 256
#define HEAD_DIM_NAMESPACE H256
#define XQA_PAGED_CACHE_ELEM 2
#define XQA_PAGED_INPUT_FP16 0
#define XQA_PAGED_QUERY_T __nv_bfloat16
#define XQA_PAGED_FAMILY bf16_fp8
#define XQA_PAGED_LAUNCH_FN LaunchXQAPagedFp8KernelBF16

#ifdef USE_FP8_KV_CACHE
#include "xqa_paged_loader_impl.cuh"
#endif
