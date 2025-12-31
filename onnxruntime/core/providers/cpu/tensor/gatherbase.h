// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#ifndef SHARED_PROVIDER
#include "core/providers/common.h"
#include "core/framework/tensor.h"
#endif

namespace onnxruntime {

class GatherBase {
 public:
  struct Prepare {
    const Tensor* input_tensor;
    const Tensor* indices_tensor;
    Tensor* output_tensor;
    int64_t axis;
  };

  template <typename KernelContextType>
  Status PrepareForComputeImpl(KernelContextType* context, Prepare& p) const {
    p.input_tensor = context->template Input<Tensor>(0);
    const TensorShape& input_data_shape = p.input_tensor->Shape();
    p.indices_tensor = context->template Input<Tensor>(1);
    const TensorShape& indices_shape = p.indices_tensor->Shape();

    const auto input_rank = input_data_shape.NumDimensions();
    p.axis = HandleNegativeAxis(axis_, narrow<int64_t>(input_rank));

    std::vector<int64_t> shape;
    shape.reserve(input_rank - 1 + indices_shape.NumDimensions());

    // replace the dimension for p.axis with the shape from the indices
    for (int64_t i = 0; i < p.axis; ++i)
      shape.push_back(input_data_shape[narrow<size_t>(i)]);

    for (const auto dim : indices_shape.GetDims())
      shape.push_back(dim);

    for (int64_t i = p.axis + 1; i < static_cast<int64_t>(input_rank); ++i)
      shape.push_back(input_data_shape[narrow<size_t>(i)]);

    p.output_tensor = context->Output(0, TensorShape(std::move(shape)));

    return Status::OK();
  }

  Status PrepareForCompute(OpKernelContext* context, Prepare& p) const;

 protected:
  template <typename KernelInfoType>
  GatherBase(const KernelInfoType& info) {
    ORT_ENFORCE(info.template GetAttr<int64_t>("axis", &axis_).IsOK(), "Missing/Invalid 'axis' attribute value");
  }

 private:
  int64_t axis_;
};

}  // namespace onnxruntime
