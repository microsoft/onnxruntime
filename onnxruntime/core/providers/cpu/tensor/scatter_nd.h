// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/narrow.h"

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#endif

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}

class ScatterND final : public OpKernel {
 public:
  enum class Reduction : int {
    None = 0,
    Add,
    Mul,
    Min,
    Max,
  };

  explicit ScatterND(const OpKernelInfo& info) : OpKernel(info) {
    // 'reduction' attribute was added in opset 16.
    // its default value is 'none' in which case the op behaves the same as before opset 16.
    std::string reduction;
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

  Status Compute(OpKernelContext* context) const override;

#ifdef SHARED_PROVIDER
  static Status ValidateShapes(const TensorShape& input_shape,
                               const TensorShape& indice_shape,
                               const TensorShape& update_shape);
#else
  static inline Status ValidateShapes(const TensorShape& input_shape,
                                      const TensorShape& indice_shape,
                                      const TensorShape& update_shape) {
    auto input_rank = input_shape.NumDimensions();
    auto indice_rank = indice_shape.NumDimensions();
    auto update_rank = update_shape.NumDimensions();

    if (input_rank == 0 || indice_rank == 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "input tensor and indices tensor must has rank larger than 0. ",
                             "input shape: ", input_shape, ", indices shape: ", indice_shape);
    }

    auto last_indice_dimension = indice_shape[indice_rank - 1];
    if (last_indice_dimension > static_cast<int64_t>(input_rank)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "last dimension of indices must not be larger than rank of input tensor");
    }

    bool is_update_shape_invalid = [&]() {
      if (update_rank != (input_rank + indice_rank - 1 - static_cast<ptrdiff_t>(last_indice_dimension))) {
        return true;
      }
      if (indice_shape.Slice(0, indice_rank - 1) != update_shape.Slice(0, indice_rank - 1)) {
        return true;
      }
      if (input_shape.Slice(onnxruntime::narrow<size_t>(last_indice_dimension)) != update_shape.Slice(indice_rank - 1)) {
        return true;
      }
      return false;
    }();

    if (is_update_shape_invalid) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "updates tensor should have shape equal to indices.shape[:-1] + data.shape[indices.shape[-1]:]. ",
                             "updates shape: ", update_shape, ", indices shape: ", indice_shape, ", data shape: ", input_shape);
    }

    return Status::OK();
  }
#endif  // SHARED_PROVIDER

 private:
  Reduction reduction_{Reduction::None};
};

}  // namespace onnxruntime
