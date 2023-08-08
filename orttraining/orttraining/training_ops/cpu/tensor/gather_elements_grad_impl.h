// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <class Tin, class Tdata>
Status GatherElementsGradImpl(const Tensor* indices_input,
                              const Tensor* updates_input,
                              const int64_t axis,
                              Tensor* data_output);

}  // namespace contrib
}  // namespace onnxruntime
