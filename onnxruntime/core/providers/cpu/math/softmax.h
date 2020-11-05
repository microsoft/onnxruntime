// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl-lite.hpp"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/softmax_shared.h"
#include "core/providers/cpu/tensor/transpose.h"
#include <vector>
#include <numeric>

namespace onnxruntime {
template <typename T>
class Softmax final : public OpKernel {
 public:
  Softmax(const OpKernelInfo& info) : OpKernel{info} {
    const auto& node = info.node();
    opset_ = node.SinceVersion();

    int64_t axis;
    Status status = info.GetAttr<int64_t>("axis", &axis);

    if (status.IsOK()) {
      axis_ = gsl::narrow_cast<int>(axis);
    } else {
      if (opset_ < 13) {
        axis_ = 1;  // opset-12 and below, the default axis value is 1
      } else {
        axis_ = -1;  // opset-13, the default axis value is -1
      }
    }

    log_softmax_ = info.GetKernelDef().OpName() == "LogSoftmax";
  }

  Status Compute(OpKernelContext* ctx) const override {
    const auto* X = ctx->Input<Tensor>(0);
    const auto& X_shape = X->Shape();
    size_t rank = X_shape.NumDimensions();
    auto* Y = ctx->Output(0, X_shape);

    // edge case. one or more dims with value of 0. nothing to do
    if (X_shape.Size() == 0) {
      return Status::OK();
    }

    const size_t axis = static_cast<size_t>(HandleNegativeAxis(axis_, rank));
    concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

    if (opset_ < 13) {
      return ComputeImpl<T>(*X, *Y, axis, thread_pool);
    } else {
      return ComputeImplOpset13<T>(*X, *Y, axis, thread_pool, ctx);
    }
  }

 private:
  template <typename T>
  Status ComputeImpl(const Tensor& input, Tensor& output, size_t axis,
                     concurrency::ThreadPool* thread_pool) const {
    const auto& X_shape = input.Shape();
    const size_t N = X_shape.SizeToDimension(axis);
    const size_t D = X_shape.SizeFromDimension(axis);

    return SoftmaxCPU<T>(N, D, input.template Data<T>(), output.template MutableData<T>(), log_softmax_, thread_pool);
  }

  template <typename T>
  Status ComputeImplOpset13(const Tensor& input, Tensor& output, int64_t axis,
                            concurrency::ThreadPool* thread_pool, OpKernelContext* ctx) const {
    const auto& X_shape = input.Shape();
    size_t rank = X_shape.NumDimensions();

    bool is_transpose_required = false;
    Tensor transposed_input;
    std::vector<int64_t> transposed_input_dims;
    Tensor intermediate_output;  // output that the softmax implementation will write into while using transposed input
    std::vector<size_t> permutation(rank);

    // The "semantic" meaning of axis has changed in opset-13.
    // Please compare: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax
    // with https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Softmax-11 for detailed explanations
    // To account for the opset-13 behavior, our plan will be to transpose the "axis" dim to the innermost dim
    // and perform softmax and then reverse the transpose. We can skip the transposing aspect if the axis is already
    // the innermost dim
    if (axis != (static_cast<int64_t>(rank) - 1)) {
      is_transpose_required = true;
    }

    if (is_transpose_required) {
      AllocatorPtr alloc;
      auto status = ctx->GetTempSpaceAllocator(&alloc);
      if (!status.IsOK())
        return status;

      std::iota(std::begin(permutation), std::end(permutation), 0);

      // swap the innermost dim with the dim corresponding to axis
      permutation[axis] = static_cast<int64_t>(rank) - 1;
      permutation[rank - 1] = axis;

      transposed_input_dims.reserve(rank);
      for (auto e : permutation) {
        transposed_input_dims.push_back(X_shape[e]);
      }

      // Allocate a temporary tensor to hold transposed input
      Tensor temp_input(input.DataType(), TensorShape(transposed_input_dims), alloc);

      // Perform the transpose
      TransposeBase::DoTranspose(permutation, input, temp_input);
      transposed_input = std::move(temp_input);

      // Allocate memory for the intermediate output
      Tensor temp_output(output.DataType(), TensorShape(transposed_input_dims), alloc);
      intermediate_output = std::move(temp_output);
    }

    const size_t N = is_transpose_required ? TensorShape(transposed_input_dims).SizeToDimension(rank - 1) : X_shape.SizeToDimension(rank - 1);
    const size_t D = is_transpose_required ? TensorShape(transposed_input_dims).SizeFromDimension(rank - 1) : X_shape.SizeFromDimension(rank - 1);

    auto status = SoftmaxCPU<T>(N, D,
                                is_transpose_required ? transposed_input.template Data<T>() : input.template Data<T>(),
                                is_transpose_required ? intermediate_output.template MutableData<T>() : output.template MutableData<T>(),
                                log_softmax_, thread_pool);
    if (!status.IsOK()) {
      return status;
    }

    if (is_transpose_required) {
      std::vector<size_t> reverse_permutation(rank);
      for (size_t i = 0; i < permutation.size(); ++i) {
        reverse_permutation[permutation[i]] = i;
      }
      // Perform the transpose to get the axes back to the original ordering
      TransposeBase::DoTranspose(reverse_permutation, intermediate_output, output);
    }

    return Status::OK();
  }

  int axis_;
  int opset_;
  bool log_softmax_;
};

}  // namespace onnxruntime
