// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/ml/ml_common.h"

#include "gsl/gsl"

namespace onnxruntime {
namespace ml {

class Normalizer final : public OpKernel {
 public:
  Normalizer(const OpKernelInfo& info) : OpKernel(info) {
    std::string norm;
    ORT_ENFORCE(info.GetAttr<std::string>("norm", &norm).IsOK());

    normalization_ = MakeNormalize(norm);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T>
  void Normalize(OpKernelContext* context) const;

  template<class>
  struct CallNormalizerImpl;

  NORMALIZE normalization_;
};

}  // namespace ml
}  // namespace onnxruntime
