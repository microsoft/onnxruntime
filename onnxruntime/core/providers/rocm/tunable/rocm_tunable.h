// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// solving ODR violation due to provider_api.h
#include "core/providers/shared_library/provider_api.h"

#include "core/framework/tunable.h"
#include "core/providers/rocm/tunable/util.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {

using OpParams = ::onnxruntime::tunable::OpParams<hipStream_t>;

template <typename ParamsT>
using Op = ::onnxruntime::tunable::Op<ParamsT>;

class Timer;

template <typename ParamsT>
using TunableOp = ::onnxruntime::tunable::TunableOp<ParamsT, ::onnxruntime::rocm::tunable::Timer>;

}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
