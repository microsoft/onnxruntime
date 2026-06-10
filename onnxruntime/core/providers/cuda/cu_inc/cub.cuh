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

// Workaround for CCCL (CUDA 13.x) header-ordering issue:
// <cub/device/device_transform.cuh> specializes cuda::proclaims_copyable_arguments,
// whose primary template lives in <cuda/__functional/address_stability.h>.
// Under some CUDA 13.x toolkits the cub umbrella reaches device_transform.cuh
// before address_stability.h, causing a cudafe++ parse error on the
// specialization (`global qualification of class name is invalid before ':'`).
// Force the primary template to be visible first.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13)
#include <cuda/__functional/address_stability.h>
#endif

#include <cub/cub.cuh>

#if defined(_MSC_VER)
#pragma pop_macro("__out")
#endif
