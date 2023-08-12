// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/gsl.h"
#include "core/providers/js/js_kernel.h"
#include "core/providers/cpu/tensor/gather.h"
#include "core/providers/cpu/tensor/gather_elements.h"
#include "core/providers/cpu/tensor/gather_nd.h"

namespace onnxruntime {
namespace js {

class Gather : public JsKernel , public GatherBase {
 public:
  Gather(const OpKernelInfo& info) : JsKernel(info), GatherBase(info) {
    int64_t axis = info.GetAttrOrDefault<int64_t>("axis", 0);

    JSEP_INIT_KERNEL_ATTRIBUTE(Gather, ({
                                 "axis" : $1
                               }),
                               gsl::narrow_cast<int32_t>(axis));
  }

};
class GatherElements : public JsKernel {
 public:
  GatherElements(const OpKernelInfo& info) : JsKernel(info) {
    int64_t axis = info.GetAttrOrDefault<int64_t>("axis", 0);
    JSEP_INIT_KERNEL_ATTRIBUTE(GatherElements, ({
                                 "axis" : $1
                               }),
                               gsl::narrow_cast<int32_t>(axis));
  }
};

class GatherND : public JsKernel, protected GatherNDBase {
 public:
  GatherND(const OpKernelInfo& info) : JsKernel(info) {
    int64_t batch_dims = info.GetAttrOrDefault<int64_t>("batch_dims", 0);
    JSEP_INIT_KERNEL_ATTRIBUTE(GatherND, ({
                                 "batch_dims" : $1
                               }),
                               gsl::narrow_cast<int32_t>(batch_dims));
  }
};

}  // namespace js
}  // namespace onnxruntime
