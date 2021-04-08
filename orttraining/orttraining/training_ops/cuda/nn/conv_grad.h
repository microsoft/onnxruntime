// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/cuda/nn/conv.h"

namespace onnxruntime {
namespace cuda {

struct ConvolutionArgs {
  cudnnHandle_t handle;
  cudnnDataType_t data_type;

  CudnnTensor i_desc, o_desc, b_desc;
  CudnnFilterDescriptor w_desc;
  CudnnConvolutionDescriptor c_desc;

  ConvolutionArgs() {}
};

template <typename T>
class ConvGrad final : public CudaKernel {
 public:
  using CudaT = typename ToCudaType<T>::MappedType;

  ConvGrad(const OpKernelInfo& info) : CudaKernel(info), conv_attrs_(info) {
#if (defined(CUDA_VERSION) && (CUDA_VERSION < 10000) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
    ORT_THROW("ConvGrad CUDA kernel is not yet tested on __CUDA_ARCH__ lower than 700");
#endif
    auto pads_size = conv_attrs_.pads.size();
    ORT_ENFORCE(pads_size % 2 == 0);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  mutable ConvolutionArgs args_;
  Status PrepareArgs(const Tensor& input, const Tensor& output, const Tensor& weight, const Tensor* bias) const;

  ConvAttributes conv_attrs_;

  // https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn_742/cudnn-developer-guide/index.html#tensor_ops
  static constexpr auto kDefaultConvBwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  static constexpr auto kDefaultConvBwdFilterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

 private:
  Status ComputeWeightGradient(Tensor* dW, const Tensor* dY, const Tensor* X) const;
  Status ComputeInputGradient(Tensor* dX, const Tensor* dY, const Tensor* W) const;
  Status ComputeBiasGradient(Tensor* dB, const Tensor* dY) const;
};

}  // namespace cuda
}  // namespace onnxruntime
