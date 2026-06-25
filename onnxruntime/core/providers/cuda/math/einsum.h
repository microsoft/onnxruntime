// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/threadpool.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/math/einsum_utils/einsum_typed_compute_processor.h"
#include "einsum_utils/einsum_auxiliary_ops.h"

namespace onnxruntime {
namespace cuda {

class Einsum final : public CudaKernel {
 public:
  explicit Einsum(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(info.GetAttr<std::string>("equation", &equation_).IsOK(),
                "Missing 'equation' attribute");
    einsum_equation_preprocessor_ = std::make_unique<EinsumEquationPreprocessor>(equation_);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  std::string equation_;
  std::unique_ptr<EinsumEquationPreprocessor> einsum_equation_preprocessor_;
};

}  // namespace cuda
}  // namespace onnxruntime
