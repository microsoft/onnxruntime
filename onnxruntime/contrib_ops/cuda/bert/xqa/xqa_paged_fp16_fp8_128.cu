// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define HEAD_ELEMS 128
#define HEAD_DIM_NAMESPACE H128
#define XQA_PAGED_CACHE_ELEM 2
#define XQA_PAGED_INPUT_FP16 1
#define XQA_PAGED_QUERY_T half
#define XQA_PAGED_FAMILY fp16_fp8
#define XQA_PAGED_LAUNCH_FN LaunchXQAPagedFp8Kernel

#ifdef USE_FP8_KV_CACHE
#include "xqa_paged_loader_impl.cuh"
#endif
