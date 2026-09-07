// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// XQA paged-KV decode kernel: FP16 query/output and FP16 KV cache, head_size=256.

#define HEAD_ELEMS 256
#define HEAD_DIM_NAMESPACE H256
#define XQA_PAGED_CACHE_ELEM 0  // FP16 KV cache
#define XQA_PAGED_INPUT_FP16 1  // FP16 query/output
#define XQA_PAGED_QUERY_T half
#define XQA_PAGED_FAMILY fp16_fp16
#define XQA_PAGED_LAUNCH_FN LaunchXQAPagedFp16Kernel
#define XQA_PAGED_GROUP6_ONLY

#include "xqa_paged_loader_impl.cuh"

#undef XQA_PAGED_GROUP6_ONLY
