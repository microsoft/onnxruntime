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

struct CudnnConvGradState : public CudnnConvState<cudnnConvolutionBwdDataAlgoPerf_t> {
  cudnnConvolutionBwdFilterAlgo_t filter_algo;
  size_t filter_workspace_bytes;
  const void* dy_data = nullptr;
  void* dx_data = nullptr;
  void* dw_data = nullptr;
  void* db_data = nullptr;

  struct FilterPerfResultParams {
    cudnnConvolutionBwdFilterAlgo_t algo;
    size_t memory;
    cudnnMathType_t mathType;
  };

  lru_unordered_map<std::vector<int64_t>, FilterPerfResultParams, vector_hash<int64_t>> filter_cached_benchmark_results{
      MAX_CACHED_ALGO_PERF_RESULTS};
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
  Status PrepareArgs(const Tensor& x, const Tensor& dY, const Tensor& w, Tensor* dB, Tensor* dX, Tensor* dW) const;
  mutable CudnnConvGradState s_;
  ConvAttributes conv_attrs_;

  // https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn_742/cudnn-developer-guide/index.html#tensor_ops
  static constexpr auto kDefaultConvBwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  static constexpr auto kDefaultConvBwdFilterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

 private:
  Status ComputeWeightGradient() const;
  Status ComputeInputGradient() const;
  Status ComputeBiasGradient() const;
};

}  // namespace cuda
}  // namespace onnxruntime
