// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "core/providers/rocm/rocm_common.h"  // avoid provider_api.h ODR violation
#include "core/framework/tunable.h"
#include "core/providers/rocm/rocm_execution_provider_info.h"
#include "core/providers/rocm/tunable/rocm_tuning_context.h"
#include "core/providers/rocm/tunable/util.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {

using OpParams = OpParams<RocmTuningContext, hipStream_t>;

template <typename ParamsT>
using Op = Op<ParamsT>;

class Timer;

template <typename ParamsT>
using TunableOp = TunableOp<ParamsT, Timer>;

}  // namespace tunable
}  // namespace rocm

// As a convenience for authoring TunableOp in contrib namespace
namespace contrib {
namespace rocm {
using onnxruntime::rocm::tunable::Op;
using onnxruntime::rocm::tunable::OpParams;
using onnxruntime::rocm::tunable::RocmTuningContext;
using onnxruntime::rocm::tunable::TunableOp;
}  // namespace rocm
}  // namespace contrib

}  // namespace onnxruntime
