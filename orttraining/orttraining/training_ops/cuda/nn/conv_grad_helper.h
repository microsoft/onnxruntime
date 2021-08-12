// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/nn/conv.h"
#include "core/platform/ort_mutex.h"

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

struct ConvParamsHash {
  size_t operator()(const ConvParams& conv_params) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&conv_params);
    uint32_t value = 0x811C9DC5;
    for (int i = 0; i < static_cast<int>(sizeof(ConvParams)); ++i) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return static_cast<size_t>(value);
  }
};

struct ConvParamsEqual {
  bool operator()(const ConvParams& a, const ConvParams& b) const {
    auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
    auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
    return memcmp(ptr1, ptr2, sizeof(ConvParams)) == 0;
  }
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
struct AlgoPerfCache {
  mutable OrtMutex mutex;
  std::unordered_map<ConvParams, T_Perf, ConvParamsHash, ConvParamsEqual> map;

  bool Find(const ConvParams& params, T_Perf* result) {
    std::lock_guard<OrtMutex> guard(mutex);
    auto it = map.find(params);
    if (it == map.end()) {
      return false;
    }
    *result = it->second;
    return true;
  }

  void Insert(const ConvParams& params, const T_Perf& algo_perf) {
    std::lock_guard<OrtMutex> guard(mutex);
    map[params] = algo_perf;
  }
};

template <typename T_Perf>
struct AlgoSearch {};

template <typename T_Perf>
class AlgoIterator {
 public:
  AlgoIterator(const ConvArgs& args) : args_(args) {}

  static Status OnlyDefaultAlgorithm(const ConvArgs& args, std::vector<T_Perf>& perf_results) {
    perf_results.resize(1);
    perf_results[0].algo = AlgoSearch<T_Perf>::DEFAULT_ALGO;
    if (args.params.data_type == CUDNN_DATA_HALF) {
      perf_results[0].mathType = CUDNN_TENSOR_OP_MATH;
    } else {
      perf_results[0].mathType = CUDNN_DEFAULT_MATH;
    }
    CUDNN_RETURN_IF_ERROR(GetWorkspaceSize(args, perf_results[0].algo, &(perf_results[0].memory)));
    return Status::OK();
  }

  Status TryAll(const CUDAExecutionProvider* provider, std::function<Status(const T_Perf& perf)> f);

 private:
  const ConvArgs& args_;
};

}  // namespace cuda
}  // namespace onnxruntime
