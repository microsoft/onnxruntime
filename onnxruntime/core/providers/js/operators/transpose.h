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
      perm[0] = gsl::narrow_cast<int32_t>(perm_.size());
      for (size_t i = 0; i < perm_.size(); ++i) {
        perm[i] = gsl::narrow_cast<int32_t>(perm_[i]);
      }
    }
    // printf("Transpose: perm_specified_ = %d, perm.size() = %d, perm[0] = %d, perm[1] = %d, perm[2] = %d, perm[3] = %d\n",
    //   perm_specified_, static_cast<int32_t>(perm.size()), perm[0], perm[1], perm[2], perm[3]);
    JSEP_INIT_KERNEL_ATTRIBUTE(Transpose, ({
                                 "perm" : $1 ? Array.from(HEAP32.subarray($2, $2 + $1)) : []
                               }),
                               gsl::narrow_cast<int32_t>(perm_specified_ ? perm_.size() : 0),
                               reinterpret_cast<int32_t>(perm_specified_ && !perm.empty() ? perm.data() : nullptr) >> 2);
  }
};

}  // namespace js
}  // namespace onnxruntime
