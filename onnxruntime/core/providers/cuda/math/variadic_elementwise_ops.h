// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <vector>

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

using InputTensorVector = std::vector<std::reference_wrapper<const Tensor>>;

template <typename VariadicElementwiseOpTag,
          typename... SupportedElementTypes>
class VariadicElementwiseOp : public CudaKernel {
 public:
  VariadicElementwiseOp(const OpKernelInfo& info) : CudaKernel(info) {}

 private:
  Status ComputeInternal(OpKernelContext* context) const override;

  template <typename T>
  struct NoBroadcastBatchImplDispatchTarget {
    Status operator()(cudaStream_t stream, const InputTensorVector& inputs, Tensor& output) const;
  };

  template <typename T>
  struct BinaryImplDispatchTarget {
    Status operator()(cudaStream_t stream, const Tensor& lhs, const Tensor& rhs, Tensor& output) const;
  };

  template <typename T>
  struct GeneralImplDispatchTarget {
    Status operator()(cudaStream_t stream, const InputTensorVector& inputs, Tensor& output) const;
  };
};

}  // namespace cuda
}  // namespace onnxruntime
