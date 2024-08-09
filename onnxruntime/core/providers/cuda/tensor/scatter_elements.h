// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

class ScatterElements final : public CudaKernel {
 public:
  ScatterElements(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(),
                "Missing/Invalid 'axis' attribute value");
    reduction_ = info.GetAttrOrDefault<std::string>("reduction", "none");

    ORT_ENFORCE(reduction_ == "none" || reduction_ == "add" ||
                    reduction_ == "mul" || reduction_ == "max" ||
                    reduction_ == "min",
                "Invalid reduction attribute value of ", reduction_);
  }
  ~ScatterElements() = default;
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  struct ComputeImpl;

  int64_t axis_;
  // "reduction" attribute has been defined since opset 13 but
  // we never implemented it. Let's try to support them starting
  // with opset 18.
  std::string reduction_;
};

}  // namespace cuda
}  // namespace onnxruntime
