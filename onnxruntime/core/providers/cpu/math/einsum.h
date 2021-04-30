// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "einsum_utils/einsum_compute_preprocessor.h"
#include "einsum_utils/einsum_typed_compute_processor.h"

namespace onnxruntime {

class Einsum : public OpKernel {
 public:
  Einsum(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<std::string>("equation", &equation_).IsOK(),
                "Missing 'equation' attribute");
    einsum_equation_preprocessor_ = std::make_unique<EinsumEquationPreprocessor>(equation_);
  }

  virtual Status Compute(OpKernelContext* context) const override;

 protected:
  // Holds device specific (CPU / CUDA) compute logic
  virtual Status DeviceCompute(OpKernelContext* context, const std::vector<const Tensor*>& inputs,
                               AllocatorPtr allocator, concurrency::ThreadPool* tp) const;

  std::string equation_;
  std::unique_ptr<EinsumEquationPreprocessor> einsum_equation_preprocessor_;
};

}  // namespace onnxruntime
