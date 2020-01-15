// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace embed_layer_norm {

Status CheckInputs(const OpKernelContext* context);

}  // namespace embed_layer_norm
}  // namespace contrib
}  // namespace onnxruntime
