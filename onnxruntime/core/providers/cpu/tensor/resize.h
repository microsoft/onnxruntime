// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/tensor/upsample.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

void UpsampleNearest2x(
    int64_t batch_size,
    int64_t num_channels,
    int64_t input_height,
    int64_t input_width,
    const float* input,
    float* output);

Status UpsampleNearest(
	const float* input,
    float* output,
    const TensorShape& input_shape,
    const TensorShape& output_shape,
    const vector<float>& scales);

void upsampleBilinear(
    int64_t batch_size,
    int64_t num_channels,
    int64_t input_height,
    int64_t input_width,
    float height_scale,
    float width_scale,
    const float* Xdata,
    float* Ydata);
	
template <typename T>
class Resize : public UpsampleBase, public OpKernel {
 public:
  Resize(OpKernelInfo info) : UpsampleBase(info), OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

  Status BaseCompute(OpKernelContext* context, const std::vector<float>& scales) const;
};

}  // namespace onnxruntime
