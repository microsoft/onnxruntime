// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#if defined(_MSC_VER)
#pragma warning(push)
//cuda_fp16.hpp(394,38): warning C4505: '__float2half_rz': unreferenced local function has been removed
#pragma warning(disable : 4505)
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cusparse.h>
#include <curand.h>
#include <cudnn.h>
#include <cufft.h>

#ifdef ORT_USE_NCCL
#include <nccl.h>
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
