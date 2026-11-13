// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_bf16.h>

#define HEAD_ELEMS 128
#define HEAD_DIM_NAMESPACE H128
#define XQA_PAGED_CACHE_ELEM 1
#define XQA_PAGED_INPUT_FP16 0
#define XQA_PAGED_QUERY_T __nv_bfloat16
#define XQA_PAGED_FAMILY spec_dec_bf16_int8_h128
#define XQA_PAGED_LAUNCH_FN LaunchXQAPagedSpecDecInt8KernelBF16
#define XQA_PAGED_GROUP6_ONLY 1
#define XQA_PAGED_SPEC_DEC 1

#ifdef _MSC_VER
#pragma warning(disable : 4459)
#endif

#include "xqa_paged_loader_impl.cuh"
