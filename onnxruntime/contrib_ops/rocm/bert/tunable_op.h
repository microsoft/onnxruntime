// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#warning "this file is moved, use core/providers/rocm/tunable/tunable.h instead!"
#include "core/providers/rocm/tunable/tunable.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using ::onnxruntime::rocm::tunable::OpParams;

template <typename ParamT>
using Op = ::onnxruntime::rocm::tunable::Op<ParamT>;

template <typename ParamT>
using TunableOp = ::onnxruntime::rocm::tunable::TunableOp<ParamT>;

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
