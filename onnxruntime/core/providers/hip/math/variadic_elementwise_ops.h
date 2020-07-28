// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <vector>

#include "core/providers/hip/hip_common.h"

namespace onnxruntime {
namespace hip {

using InputTensorVector = std::vector<std::reference_wrapper<const Tensor>>;

template <typename VariadicElementwiseOpTag,
          typename... SupportedElementTypes>
class VariadicElementwiseOp : public HipKernel {
 public:
  VariadicElementwiseOp(const OpKernelInfo& info) : HipKernel(info) {}

 private:
  Status ComputeInternal(OpKernelContext* context) const override;

  template <typename T>
  struct NoBroadcastBatchImplDispatchTarget {
    Status operator()(const InputTensorVector& inputs, Tensor& output) const;
  };

  template <typename T>
  struct BinaryImplDispatchTarget {
    Status operator()(const HipKernel* kernel, const Tensor& lhs, const Tensor& rhs, Tensor& output) const;
  };

  template <typename T>
  struct GeneralImplDispatchTarget {
    Status operator()(const HipKernel* kernel, const InputTensorVector& inputs, Tensor& output) const;
  };
};

}  // namespace hip
}  // namespace onnxruntime
