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

constexpr int max_dim = 3;

// This POD struct is used to let us easily compute hashes of the
// parameters
struct ConvolutionParams {
  // c10::DeviceIndex device_id;
  cudnnDataType_t dataType;
  int input_size[2 + max_dim];
  uint8_t input_dim;
  // at::MemoryFormat memory_format;
  int weight_size[2 + max_dim];
  int padding[max_dim];
  int stride[max_dim];
  int dilation[max_dim];
  int64_t groups;
  bool deterministic;
  bool allow_tf32;
  // NB: transposed purposely omitted: transposed just swaps
  // forward and backward, so you can reuse the benchmark entry,
};

// NB: This can't be a constructor, because then ConvolutionParams
// would not be a POD anymore.
// TODO: Use TensorGeometry here instead of the entire Tensor, which we
// don't actually need.  (OTOH: We can always pass in
// grad_input/grad_output, so this is not very pressing)
void setConvolutionParams(
    ConvolutionParams* params,
    const at::Tensor& input, const at::Tensor& weight,
    const std::vector<int64_t>& padding, const std::vector<int64_t>& stride, const std::vector<int64_t>& dilation,
    int64_t groups, bool deterministic, bool allow_tf32);

struct ConvolutionArgs {
  cudnnHandle_t handle;
  ConvolutionParams params;

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

  Status getBwdDataAlgoPerf(const std::vector<int64_t>& x_dims, const ConvolutionArgs& args, int cudnn_conv_algo_search,
                            const void* w, const void* dy, void* dx, cudnnConvolutionBwdDataAlgoPerf_t& perf) const;

  Status getBwdFilterAlgoPerf(const std::vector<int64_t>& x_dims, const ConvolutionArgs& args, int cudnn_conv_algo_search,
                              const void* x, const void* dy, void* dw, cudnnConvolutionBwdFilterAlgoPerf_t& perf) const;

  // static const cudnnConvolutionBwdDataAlgo_t kAllBwdDataAlgo[];
  // static const cudnnConvolutionBwdFilterAlgo_t kAllBwdFilterAlgo[];

 private:
  Status ComputeWeightGradient(Tensor* dW, const Tensor* dY, const Tensor* X) const;
  Status ComputeInputGradient(Tensor* dX, const Tensor* dY, const Tensor* W) const;
  Status ComputeBiasGradient(Tensor* dB, const Tensor* dY) const;
};

}  // namespace cuda
}  // namespace onnxruntime
