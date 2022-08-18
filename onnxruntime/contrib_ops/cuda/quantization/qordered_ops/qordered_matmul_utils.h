// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

#include <string>
#include <unordered_set>

namespace onnxruntime {

namespace contrib {
namespace cuda {

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
    int algo_id, custom_option, tile, splitk_val,
        swizzle, reduction_scheme, workspace_size, stages;

    float exec_time;
  };

  std::unordered_map<std::string, CublasLtMatmulAlgoInfo> best_algos_;
};

Status QOrdered_MatMul(cublasLtHandle_t cublasLt_handle, cudaStream_t stream,
                       const cudaDeviceProp& device_prop,
                       int32_t batch_count, int64_t m, int64_t n, int64_t k,
                       const float* alpha, const int8_t* A, const int8_t* B,
                       int32_t batch_B,
                       const float* bias, const float* beta,
                       const int8_t* C, int32_t batch_C,
                       int8_t* D, cublasLtOrder_t weight_order);

Status QOrdered_MatMul(cublasLtHandle_t cublasLt_handle, cudaStream_t stream,
                       const cudaDeviceProp& device_prop,
                       int32_t batch_count, int64_t m, int64_t n, int64_t k,
                       const float* alpha, const int8_t* A, const int8_t* B,
                       const float* bias, int8_t* D, cublasLtOrder_t order_weight);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
