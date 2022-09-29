// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/miopen_common.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/rocm/nn/conv.h"

namespace onnxruntime {
namespace rocm {

constexpr int MAX_DIM = 3;

struct ConvParams {
  int8_t device_id;
  miopenDataType_t data_type;
  int input_size[2 + MAX_DIM];
  uint8_t input_dim;
  int weight_size[2 + MAX_DIM];
  int padding[MAX_DIM * 2];
  int stride[MAX_DIM];
  int dilation[MAX_DIM];
  int64_t groups;
};

struct ConvArgs {
  // Update needed if x or w's dims changed.
  TensorShapeVector last_x_dims;
  TensorShapeVector last_w_dims;

  miopenHandle_t handle;
  ConvParams params;
  MiopenTensor x_tensor, y_tensor, b_tensor;
  MiopenTensorDescriptor w_desc;
  MiopenConvolutionDescriptor conv_desc;
  const void* x_data;
  const void* w_data;
  const void* dy_data;
  void* dx_data;
  void* dw_data;
  void* db_data;
};

template <typename T>
class ConvGrad final : public RocmKernel {
 public:
  using HipT = typename ToHipType<T>::MappedType;

  ConvGrad(const OpKernelInfo& info) : RocmKernel(info), conv_attrs_(info) {
    auto pads_size = conv_attrs_.pads.size();
    ORT_ENFORCE(pads_size % 2 == 0);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  Status PrepareArgs(const Tensor& x, const Tensor& dY, const Tensor& w, Tensor* dB, Tensor* dX, Tensor* dW) const;
  mutable ConvArgs args_;
  ConvAttributes conv_attrs_;

 private:
  Status ComputeWeightGradient() const;
  Status ComputeInputGradient() const;
  Status ComputeBiasGradient() const;
};

}  // namespace rocm
}  // namespace onnxruntime
