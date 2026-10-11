// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define HEAD_ELEMS 256
#define HEAD_DIM_NAMESPACE H256
#define XQA_PAGED_CACHE_ELEM 1
#define XQA_PAGED_INPUT_FP16 1
#define XQA_PAGED_QUERY_T half
#define XQA_PAGED_FAMILY fp16_int8
#define XQA_PAGED_LAUNCH_FN LaunchXQAPagedInt8Kernel

#include "xqa_paged_loader_impl.cuh"
