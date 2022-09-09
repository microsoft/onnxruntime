// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#if defined(_MSC_VER)
#pragma warning(push)
//hip_fp16.hpp(394,38): warning C4505: '__float2half_rz': unreferenced local function has been removed
#pragma warning(disable : 4505)
#endif

#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <hipsparse.h>
#include <hiprand/hiprand.h>
#include <miopen/miopen.h>
#include <hipfft.h>

#ifdef ORT_USE_NCCL
#include <rccl.h>
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
