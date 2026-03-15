// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Wrapper include for <cub/cub.cuh>.

// Macro definition of `__out` (SAL annotation) conflicts with parameter name `__out` in
// <CUDA v13.1 directory>/include/cccl/cuda/__ptx/instructions/generated/tcgen05_ld.h.
// As a workaround, undefine `__out` before including cub/cub.cuh.
#if defined(_MSC_VER)
#pragma push_macro("__out")
#undef __out
#endif

#include <cub/cub.cuh>

#if defined(_MSC_VER)
#pragma pop_macro("__out")
#endif
