// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::js::JsKernel;
JSEP_KERNEL_IMPL(BiasSplitGelu, BiasSplitGelu);

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
