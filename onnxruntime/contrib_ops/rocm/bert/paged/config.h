// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// See https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.1.0/reference/system-requirements.html
// for llvm target code
#if defined(__gfx942__) || defined(__gfx90a__) || defined(__gfx908__)
#define PAGED_INNER_PRODUCT_FP16_ARITHMETIC_FP32_ACC 1
#else
#define PAGED_INNER_PRODUCT_FP16_ARITHMETIC_FP32_ACC 0
#endif

#define PAGED_LITTLE_ENDIAN_BIT_TEST_IS_INF_OR_NAN 0
