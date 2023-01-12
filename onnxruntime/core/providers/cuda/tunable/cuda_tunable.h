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

using OpParams = OpParams<cudaStream_t>;

template <typename ParamsT>
using Op = Op<ParamsT>;

class Timer;

template <typename ParamsT>
using TunableOp = TunableOp<ParamsT, Timer>;

}  // namespace tunable
}  // namespace cuda

// As a convenience for authoring TunableOp in contrib namespace
namespace contrib {
namespace cuda {
using onnxruntime::cuda::tunable::Op;
using onnxruntime::cuda::tunable::OpParams;
using onnxruntime::cuda::tunable::TunableOp;
}  // namespace cuda
}  // namespace contrib

}  // namespace onnxruntime
