// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <vector>

#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {

using InputTensorVector = std::vector<std::reference_wrapper<const Tensor>>;

template <typename VariadicElementwiseOpTag,
          typename... SupportedElementTypes>
class VariadicElementwiseOp : public RocmKernel {
 public:
  VariadicElementwiseOp(const OpKernelInfo& info) : RocmKernel(info) {}

 private:
  Status ComputeInternal(OpKernelContext* context) const override;

  template <typename T>
  struct NoBroadcastBatchImplDispatchTarget {
    Status operator()(const InputTensorVector& inputs, Tensor& output) const;
  };

  template <typename T>
  struct BinaryImplDispatchTarget {
    Status operator()(const RocmKernel* kernel, const Tensor& lhs, const Tensor& rhs, Tensor& output) const;
  };

  template <typename T>
  struct GeneralImplDispatchTarget {
    Status operator()(const RocmKernel* kernel, const InputTensorVector& inputs, Tensor& output) const;
  };
};

}  // namespace rocm
}  // namespace onnxruntime
