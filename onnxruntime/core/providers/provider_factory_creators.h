// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_execution_provider_info.h"
#endif
#ifdef USE_ROCM
#include "core/providers/rocm/rocm_execution_provider_info.h"
#endif

// declarations of internal execution provider factory creation functions
// TODO centralize more of the CreateExecutionProviderFactory_X() declarations here
namespace onnxruntime {

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CPU(int use_arena);

#ifdef USE_CUDA
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(const CUDAExecutionProviderInfo& info);
#endif

#ifdef USE_ROCM
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ROCM(const ROCMExecutionProviderInfo& info);
#endif
}  // namespace onnxruntime
