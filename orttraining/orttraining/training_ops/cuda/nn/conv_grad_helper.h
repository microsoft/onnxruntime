// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/nn/conv.h"

namespace onnxruntime {
namespace cuda {

// cuDNN only takes 4D or 5D x tensor.
constexpr int MAX_DIM = 3;

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
  std::vector<int64_t> last_x_dims;
  std::vector<int64_t> last_w_dims;

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

template <typename T_Perf>
class AlgoIterator {
 public:
  AlgoIterator(const ConvArgs& args) : args_(args) {}
  static Status OnlyDefaultAlgorithm(const ConvArgs& args, std::vector<T_Perf>& perf_results);
  Status TryAll(const CUDAExecutionProvider* provider, std::function<Status(const T_Perf& perf)> f);

 private:
  const ConvArgs& args_;
};

}  // namespace cuda
}  // namespace onnxruntime
