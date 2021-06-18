// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/framework/tensor.h"
#endif

namespace onnxruntime {

class UnsqueezeBase {
 public:
  struct Prepare {
    const Tensor* input_tensor = nullptr;
    Tensor* output_tensor = nullptr;
  };

  Status PrepareCompute(OpKernelContext* context, Prepare& p) const;

 protected:
  UnsqueezeBase(const OpKernelInfo& info) {
    size_t num_inputs = info.GetInputCount();
    if (num_inputs == 1) {  //axes must be a valid attribute
      ORT_ENFORCE(info.GetAttrs("axes", axes_).IsOK(), "Missing/Invalid 'axes' attribute value");
    }
  }

 private:
  std::vector<int64_t> axes_;
};

class Unsqueeze final : public OpKernel, public UnsqueezeBase {
 public:
  Unsqueeze(const OpKernelInfo& info) : OpKernel(info), UnsqueezeBase(info) {}
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime
