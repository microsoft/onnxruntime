// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/tensor/scatter_nd.h"

namespace onnxruntime {
namespace cuda {

class ScatterND final : public CudaKernel {
  // TODO: this is common to cpu/cuda providers, where should it be placed?
  enum class Reduction : int {
    None = 0,
    Add = 1,
    Mul = 2,
    Min = 3,
    Max = 4,
  };  
 public:
  explicit ScatterND(const OpKernelInfo& info) : CudaKernel(info) {
    std::string reduction = info.GetAttrOrDefault<std::string>("reduction", "none");

    if (info.GetAttr<std::string>("reduction", &reduction).IsOK()) {
      if (reduction == "add")
        reduction_ = Reduction::Add;
      else if (reduction == "mul")
        reduction_ = Reduction::Mul;
      else if (reduction == "min")
        reduction_ = Reduction::Min;
      else if (reduction == "max")
        reduction_ = Reduction::Max;
    }
  }
  Status ComputeInternal(OpKernelContext* context) const override;
  
 private:
  // "reduction" attribute has been defined since opset 13 but
  // we never implemented it. Let's try to support them starting
  // with opset 18.
  Reduction reduction_{Reduction::None};
};

}  // namespace cuda
}  // namespace onnxruntime
