// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class GatherND : public JsKernel {
 public:
  GatherND(const OpKernelInfo& info) : JsKernel(info) {
    int64_t batchDims = info.GetAttrOrDefault<int64_t>("batch_dims", 0);

    JSEP_INIT_KERNEL_ATTRIBUTE(GatherND, ({
                                 "batch_dims" : Number($1),
                               }),
                               static_cast<int32_t>(batchDims));
  }
};

}  // namespace js
}  // namespace onnxruntime
