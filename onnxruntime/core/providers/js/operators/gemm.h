// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

template <typename T>
class Gemm : public JsKernel {
 public:
  Gemm(const OpKernelInfo& info) : JsKernel(info) {
    float alpha = info.GetAttrOrDefault<float>("alpha", 1.0f);
    float beta = info.GetAttrOrDefault<float>("beta", 1.0f);
    int64_t transA = info.GetAttrOrDefault<int64_t>("transA", 0);
    int64_t transB = info.GetAttrOrDefault<int64_t>("transB", 0);

    // currently only support Conv2D. TODO: support other
    JSEP_INIT_KERNEL_ATTRIBUTE(Gemm, ({
                                 "alpha" : $1,
                                 "beta" : $2,
                                 "transA" : $3,
                                 "transB" : $4
                               }),
                               static_cast<double>(alpha),
                               static_cast<double>(beta),
                               static_cast<int32_t>(transA),
                               static_cast<int32_t>(transB));
  }
};

}  // namespace js
}  // namespace onnxruntime
