// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define HEAD_ELEMS 256
#define HEAD_DIM_NAMESPACE H256
#define XQA_PAGED_CACHE_ELEM 0
#define XQA_PAGED_INPUT_FP16 0
#define XQA_PAGED_QUERY_T __nv_bfloat16
#define XQA_PAGED_FAMILY bf16_bf16_spec_dec
#define XQA_PAGED_LAUNCH_FN LaunchXQAPagedSpecDecBf16Kernel
#define XQA_PAGED_GROUP6_ONLY 1
#define XQA_PAGED_SPEC_DEC 1

#include "xqa_paged_loader_impl.cuh"