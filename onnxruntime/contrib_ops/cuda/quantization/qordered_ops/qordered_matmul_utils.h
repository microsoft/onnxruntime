// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

#include <string>
#include <unordered_set>

namespace onnxruntime {

namespace contrib {
namespace cuda {

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11040

class CublasLtMMAlgoMap {
 public:
  static CublasLtMMAlgoMap& Instance();

  void GetAlgo(cublasLtHandle_t cublasLt_handle, cublasLtMatmulAlgo_t& algo,
               const cudaDeviceProp& device_prop,
               int batch_count, int m, int n, int k,
               cublasLtOrder_t weight_order,
               cublasLtOrder_t input_output_order = CUBLASLT_ORDER_ROW) const;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CublasLtMMAlgoMap);

 private:
  CublasLtMMAlgoMap() = default;

  ~CublasLtMMAlgoMap() = default;

  struct CublasLtMatmulAlgoInfo {
    int algo_id;
    int custom_option;
    int tile;
    int splitk_val;
    int swizzle;
    int reduction_scheme;
    int workspace_size;
    int stages;

    float exec_time;
  };

  std::unordered_map<std::string, CublasLtMatmulAlgoInfo> best_algos_;
};

Status QOrdered_MatMul(cublasLtHandle_t cublasLt_handle, cudaStream_t stream,
                       const cudaDeviceProp& device_prop,
                       int32_t batch_count, int64_t m, int64_t n, int64_t k,
                       const float* alpha, const int8_t* A, const int8_t* B, int32_t batch_B,
                       const float* bias,
                       const float* beta, const int8_t* C, int32_t batch_C,
                       int8_t* D,
                       cublasLtOrder_t weight_order,
                       cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_HOST);

inline Status QOrdered_MatMul(cublasLtHandle_t cublasLt_handle, cudaStream_t stream,
                              const cudaDeviceProp& device_prop,
                              int32_t batch_count, int64_t m, int64_t n, int64_t k,
                              const float* alpha, const int8_t* A, const int8_t* B,
                              const float* bias,
                              int8_t* D,
                              cublasLtOrder_t order_weight,
                              cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_HOST) {
  return QOrdered_MatMul(cublasLt_handle, stream, device_prop,
                         batch_count, m, n, k,
                         alpha, A, B, 1,
                         bias,
                         (const float*)nullptr, nullptr, 1,
                         D,
                         order_weight,
                         pointer_mode);
}

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
