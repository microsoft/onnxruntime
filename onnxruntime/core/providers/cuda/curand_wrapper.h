//
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#if defined(CUDA_VERSION) && CUDA_VERSION == 13000
#define __NV_NO_VECTOR_DEPRECATION_DIAG 1
#endif

#include <curand_kernel.h>
