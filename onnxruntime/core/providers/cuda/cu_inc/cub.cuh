// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Wrapper include for <cub/cub.cuh>.

// Macro definition of `__out` (SAL annotation) conflicts with parameter name `__out` in
// <CUDA directory>/include/cccl/cuda/__ptx/instructions/generated/tcgen05_ld.h (and the
// other tcgen05_*.h PTX instruction headers). As a workaround, undefine `__out` before
// including the CCCL headers.
#if defined(_MSC_VER)
#pragma push_macro("__out")
#undef __out
// CCCL cub/config.cuh has a #pragma warning(pop) without matching push in CUDA v13.3.
#pragma warning(push)
#pragma warning(disable : 4193)

// Depending on the CUDA toolkit version, the CCCL PTX instruction headers are not always
// reached through <cub/cub.cuh> (e.g. they may be pulled in later by another CUDA header,
// after the macro has been restored below). Parse the PTX umbrella here, while `__out` is
// undefined, so those headers are processed safely regardless of include order. Guarded by
// __has_include so toolkits without <cuda/ptx> are unaffected.
#if defined(__has_include)
#if __has_include(<cuda/ptx>)
#include <cuda/ptx>
#endif
#endif
#endif

#include <cub/cub.cuh>

#if defined(_MSC_VER)
#pragma warning(pop)
#pragma pop_macro("__out")
#endif
