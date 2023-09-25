// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"
#include "core/common/gsl.h"
#include "core/providers/cpu/tensor/transpose.h"

namespace onnxruntime {
namespace js {

class Transpose final : public JsKernel, public TransposeBase {
 public:
  Transpose(const OpKernelInfo& info) : JsKernel(info), TransposeBase(info) {
    std::vector<int32_t> perm;
    if (perm_specified_) {
      perm.resize(perm_.size());
      for (size_t i = 0; i < perm_.size(); ++i) {
        perm[i] = gsl::narrow_cast<int32_t>(perm_[i]);
      }
    }
    JSEP_INIT_KERNEL_ATTRIBUTE(Transpose, ({
                                 "perm" : $1 ? Array.from(HEAP32.subarray($2, $2 + $1)) : []
                               }),
                               // $1: length of attribute "perm" (int32[])
                               gsl::narrow_cast<int32_t>(perm_specified_ ? perm_.size() : 0),
                               // $2: index to HEAP32 of the first int32 element. calculated from right shift memory
                               //     address by 2
                               reinterpret_cast<int32_t>(perm_specified_ && !perm.empty() ? perm.data() : nullptr) >> 2);
  }
};

}  // namespace js
}  // namespace onnxruntime
