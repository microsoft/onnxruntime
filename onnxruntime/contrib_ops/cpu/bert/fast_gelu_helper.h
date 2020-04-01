// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace fast_gelu {
Status CheckInputs(const OpKernelContext* context);

}  // namespace fast_gelu
}  // namespace contrib
}  // namespace onnxruntime
