// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime_api.h>

// solving ODR violation due to provider_api.h
#include "core/providers/shared_library/provider_api.h"

#include "core/framework/tunable.h"
#include "core/providers/cuda/tunable/util.h"

namespace onnxruntime {
namespace cuda {
namespace tunable {

using OpParams = ::onnxruntime::tunable::OpParams<cudaStream_t>;

template <typename ParamsT>
using Op = ::onnxruntime::tunable::Op<ParamsT>;

class Timer;

template <typename ParamsT>
using TunableOp = ::onnxruntime::tunable::TunableOp<ParamsT, ::onnxruntime::cuda::tunable::Timer>;

}  // namespace tunable
}  // namespace cuda
}  // namespace onnxruntime
