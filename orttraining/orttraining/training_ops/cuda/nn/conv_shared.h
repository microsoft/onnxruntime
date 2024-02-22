// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/nn/conv.h"

// The AlgoPerfCache and AlgoSearch here for Conv/ConvGrad/ConvTransposeGrad is adapted from PyTorch's implementation
// in aten/src/ATen/native/cudnn/Conv_v7.cpp.

namespace onnxruntime::cuda {

using T_BwdDataPerf = cudnnConvolutionBwdDataAlgoPerf_t;
using T_BwdDataAlgo = cudnnConvolutionBwdDataAlgo_t;
using T_BwdFilterPerf = cudnnConvolutionBwdFilterAlgoPerf_t;
using T_BwdFilterAlgo = cudnnConvolutionBwdFilterAlgo_t;
using T_FwdAlgo = cudnnConvolutionFwdAlgo_t;
using T_FwdPerf = cudnnConvolutionFwdAlgoPerf_t;

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
  TensorShapeVector last_x_dims;  // Input to the convolution
  TensorShapeVector last_w_dims;  // Weights of the convolution

  cudnnHandle_t handle;
  ConvParams params;
  CudnnTensor x_tensor, y_tensor, b_tensor;
  CudnnFilterDescriptor w_desc;
  CudnnConvolutionDescriptor conv_desc;
  const void* x_data;
  const void* w_data;
  const void* dy_data;
  void* y_data;
  void* dx_data;
  void* dw_data;
  void* db_data;
};

struct ConvParamsHash {
  // ConvParams must be a POD because we read out its memory constant as char* when hashing.
  static_assert(std::is_pod<ConvParams>::value, "ConvParams is not POD");

  size_t operator()(const ConvParams& conv_params) const;
};

struct ConvParamsEqual {
  // ConvParams must be a POD because we read out its memory constant as char* when hashing.
  static_assert(std::is_pod<ConvParams>::value, "ConvParams is not POD");

  bool operator()(const ConvParams& a, const ConvParams& b) const;
};

template <typename T_Perf>
class AlgoIterator {
 public:
  AlgoIterator(const ConvArgs& args) : args_(args) {}

  Status TryAll(const CUDAExecutionProvider* provider, const AllocatorPtr& allocator,
                std::function<Status(const T_Perf& perf)> f);

  static Status OnlyDefaultAlgorithm(const ConvArgs& args, std::vector<T_Perf>& perf_results, bool use_tf32);

 private:
  const ConvArgs& args_;
};

}  // namespace onnxruntime::cuda
