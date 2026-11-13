// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define HEAD_ELEMS 128
#define HEAD_DIM_NAMESPACE H128
#define XQA_SPEC_INPUT_FP16 0
#define XQA_SPEC_QUERY_T __nv_bfloat16
#define XQA_SPEC_FAMILY bf16
#define XQA_SPEC_LAUNCH_FN LaunchXQASpecDecBf16Kernel

// SPEC_DEC helper parameters reuse the compile-time XQA constant names; MSVC reports them as C4459.
#ifdef _MSC_VER
#pragma warning(disable : 4459)
#endif

#include "xqa_spec_dec_impl.cuh"
