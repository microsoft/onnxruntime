// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/cuda/nn/conv.h"

namespace onnxruntime {
namespace cuda {

// cuDNN only takes 4D or 5D x tensor.
static constexpr int MAX_DIM = 3;

struct ConvParams {
  int8_t device_id;
  cudnnDataType_t data_type;
  int input_size[2 + MAX_DIM];
  uint8_t input_dim;
  int weight_size[2 + MAX_DIM];
  int padding[MAX_DIM * 2];
  int stride[MAX_DIM];
  int dilation[MAX_DIM];
  int64_t groups;
  int algo_mode;
};

struct ConvArgs {
  // Update needed if x or w's dims changed.
  TensorShapeVector last_x_dims;
  TensorShapeVector last_w_dims;

  cudnnHandle_t handle;
  ConvParams params;
  CudnnTensor x_tensor, y_tensor, b_tensor;
  CudnnFilterDescriptor w_desc;
  CudnnConvolutionDescriptor conv_desc;
  const void* x_data;
  const void* w_data;
  const void* dy_data;
  void* dx_data;
  void* dw_data;
  void* db_data;
};

template <typename T>
class ConvGrad final : public CudaKernel {
 public:
  using CudaT = typename ToCudaType<T>::MappedType;

  ConvGrad(const OpKernelInfo& info) : CudaKernel(info), conv_attrs_(info) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
    ORT_THROW("ConvGrad CUDA kernel is not yet tested on __CUDA_ARCH__ lower than 700");
#endif
    auto pads_size = conv_attrs_.pads.size();
    ORT_ENFORCE(pads_size % 2 == 0);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  Status PrepareArgs(const Tensor& x, const Tensor& dY, const Tensor& w, Tensor* dB, Tensor* dX, Tensor* dW, cudnnHandle_t cudnn_handle) const;
  mutable ConvArgs args_;
  ConvAttributes conv_attrs_;

 private:
  Status ComputeWeightGradient(onnxruntime::Stream* stream) const;
  Status ComputeInputGradient(onnxruntime::Stream* stream) const;
  Status ComputeBiasGradient() const;
};

}  // namespace cuda
}  // namespace onnxruntime
