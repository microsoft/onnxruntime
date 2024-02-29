// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/tensor/scatter_nd.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using onnxruntime::OpKernelContext;
using onnxruntime::OpKernelInfo;
using onnxruntime::cuda::CudaKernel;

/**
 * This is an implementation derived from the first one.
 * It does atomic operation to handle conflicts.
 * The result is unlikely to be correct if the reduction is none
 * as there is no guarantee that the final value will be the one
 * corresponding to the highest visited index.
 * TODO: change the implementation of avoid conflicts.
 */
class ScatterNDOfShape final : public CudaKernel {
  enum class Reduction : int {
    None = 0,
    Add = 1,
    Mul = 2,
    Min = 3,
    Max = 4,
  };

 public:
  ScatterNDOfShape(const OpKernelInfo& info) : CudaKernel(info) {
    std::string reduction = info.GetAttrOrDefault<std::string>("reduction", "add");

    if (info.GetAttr<std::string>("reduction", &reduction).IsOK()) {
      if (reduction == "add")
        reduction_ = Reduction::Add;
      else if (reduction == "none") {
        LOGS_DEFAULT(WARNING) << "ScatterND with reduction=='none' only garuantees to be correct if indices are not duplicated.";
      }
      else
      ORT_THROW("Reduction '", reduction, "' is not supported on CUDA and opset >= 13.");
    }
  }
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Reduction reduction_{Reduction::Add};
};

}  // namespace cuda
}  // contrib
}  // namespace onnxruntime
