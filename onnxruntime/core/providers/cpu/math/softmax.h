/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/util/math_cpuonly.h"
#include "core/util/softmax.h"
#include "core/providers/common.h"

namespace onnxruntime {

template <typename T, bool use_log>
class Softmax final : public OpKernel {
 public:
  Softmax(const OpKernelInfo& info) : OpKernel{info}, axis_{1} {
    int64_t axis;
    Status status = info.GetAttr<int64_t>("axis", &axis);

    if (status.IsOK()) {
      axis_ = gsl::narrow_cast<int>(axis);
    }
  }

  Status Compute(OpKernelContext* ctx) const override {
#ifndef USE_OPENMP
    concurrency::ThreadPool* tp = ctx->GetOperatorThreadPool();
#endif
    const auto* tensor_pointer = ctx->Input<Tensor>(0);
    if (tensor_pointer == nullptr)
      return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
    const Tensor& X = *tensor_pointer;
    const TensorShape& input_shape = X.Shape();

    VLOGS(ctx->Logger(), 2) << "Input tensor shape: " << input_shape;

    Tensor* Y = ctx->Output(0, input_shape);

    // edge case. one or more dims with value of 0. nothing to do
    if (input_shape.Size() == 0)
      return Status::OK();

    const int64_t axis = HandleNegativeAxis(axis_, input_shape.NumDimensions());

    int N = static_cast<int>(input_shape.SizeToDimension(axis));
    int D = static_cast<int>(input_shape.SizeFromDimension(axis));

    Eigen::TensorMap<Eigen::Tensor<const float, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> X_tensor(
        X.Data<float>(), N, D);
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> Y_tensor(
        Y->MutableData<float>(), N, D);
#ifndef USE_OPENMP
    if (tp == nullptr)
#endif
      ComputeSoftMax<use_log>(Eigen::DefaultDevice(), X_tensor, Y_tensor, N, D);
#ifndef USE_OPENMP
    else
      ComputeSoftMax<use_log>(Eigen::ThreadPoolDevice(&tp->GetHandler(), tp->NumThreads()), X_tensor, Y_tensor, N, D);
#endif
    return Status::OK();
  }

 private:
  int axis_;
};
}  // namespace onnxruntime
