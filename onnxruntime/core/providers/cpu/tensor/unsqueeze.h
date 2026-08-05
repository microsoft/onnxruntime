// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/providers/common.h"
#endif

#include "utils.h"

namespace onnxruntime {

class UnsqueezeBase {
 public:
  struct Prepare {
    const Tensor* input_tensor = nullptr;
    Tensor* output_tensor = nullptr;
  };

#ifdef SHARED_PROVIDER
  Status PrepareCompute(OpKernelContext* context, Prepare& p) const;
#else
  template <typename KernelContextType>
  inline Status PrepareCompute(KernelContextType* ctx, Prepare& p) const {
    const auto* X = ctx->template Input<Tensor>(0);
    ORT_ENFORCE(X != nullptr);
    auto& input_tensor = *X;

    TensorShapeVector axes;
    size_t num_inputs = ctx->InputCount();
    if (num_inputs == 2) {
      const Tensor* axes_tensor = ctx->template Input<Tensor>(1);
      ORT_ENFORCE(axes_tensor != nullptr, "Axes input is null");
      ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 0 ||
                      axes_tensor->Shape().NumDimensions() == 1,
                  "An axes tensor must be a scalar or a 1-D tensor.");
      auto data_span = axes_tensor->template DataAsSpan<int64_t>();
      axes.assign(data_span.begin(), data_span.end());
    } else {
      axes.assign(axes_.begin(), axes_.end());
    }

    TensorShapeVector output_dims(axes.size() + input_tensor.Shape().NumDimensions(), 0);

    for (int64_t axis : axes) {
      axis = HandleNegativeAxis(axis, onnxruntime::narrow<int64_t>(output_dims.size()));
      if (axis < 0 || axis >= static_cast<int64_t>(output_dims.size()))
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "'axes' has an out of range axis");
      if (output_dims[onnxruntime::narrow<size_t>(axis)] != 0)
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "'axes' has a duplicate axis");
      output_dims[onnxruntime::narrow<size_t>(axis)] = 1;
    }

    {
      auto begin = input_tensor.Shape().GetDims().begin();
      for (auto& axis_size : output_dims) {
        if (axis_size == 0)
          axis_size = *begin++;
      }
      assert(begin == input_tensor.Shape().GetDims().end());
    }

    TensorShape output_shape(output_dims);
    p.output_tensor = ctx->Output(0, output_shape);
    ORT_ENFORCE(nullptr != p.output_tensor);
    p.input_tensor = &input_tensor;
    return Status::OK();
  }
#endif

  static TensorShapeVector ComputeOutputShape(
      const TensorShape& input_shape,
      const TensorShapeVector& axes) {
    TensorShapeVector output_shape;
    auto num_dimensions = input_shape.NumDimensions();

    auto total_num_dimensions = num_dimensions + axes.size();
    // Handle negtive axis, then resort and uniq.
    TensorShapeVector axes_corrected(axes.size());
    for (size_t i = 0; i < axes.size(); i++) {
      axes_corrected[i] = HandleNegativeAxis(axes[i], total_num_dimensions);
    }
    std::sort(axes_corrected.begin(), axes_corrected.end());
    axes_corrected.erase(std::unique(axes_corrected.begin(), axes_corrected.end()), axes_corrected.end());
    ORT_ENFORCE(axes_corrected.size() == axes.size(), "Axes input has duplicate values.");
    ORT_ENFORCE(axes_corrected.size() > 0, "Unsqueeze axes is empty.");
    size_t corr = 0;
    size_t j = 0;
    for (size_t i = 0; i < total_num_dimensions; ++i) {
      if (j < axes_corrected.size() && axes_corrected[j] == static_cast<int64_t>(i)) {
        output_shape.push_back(1);
        ++j;
        ++corr;
        continue;
      }
      output_shape.push_back(input_shape[i - corr]);
    }
    return output_shape;
  }

 protected:
  template <typename KernelInfoType>
  UnsqueezeBase(const KernelInfoType& info) {
    size_t num_inputs = info.GetInputCount();
    if (num_inputs == 1) {  // axes must be a valid attribute
      ORT_ENFORCE(info.GetAttrs("axes", axes_).IsOK(), "Missing/Invalid 'axes' attribute value");
    }
  }

  TensorShapeVector axes_;
};

class Unsqueeze final : public OpKernel, public UnsqueezeBase {
 public:
  Unsqueeze(const OpKernelInfo& info) : OpKernel(info), UnsqueezeBase(info) {}
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime
