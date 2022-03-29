// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <unordered_map>
#include <string>

namespace onnxruntime {
namespace contrib {
namespace cuda {

// #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000

using namespace onnxruntime::cuda;

class QuantizeWithOrder final : public CudaKernel {
 public:
  QuantizeWithOrder(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int order_input_;
  int order_output_;
};

class DequantizeWithOrder final : public CudaKernel {
 public:
  DequantizeWithOrder(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int order_input_;
  int order_output_;
};

class QOrderedMatMul final : public CudaKernel {
 public:
  QOrderedMatMul(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int order_A_;
  int order_B_;
  int order_Y_;
};

cublasLtOrder_t GetCublasLtOrderAttr(const OpKernelInfo& info, const char* order_attr);
cublasLtOrder_t GetCublasLtOrderAttr(const OpKernelInfo& info, const char* order_attr,
                                     int num_allowed_orders, const cublasLtOrder_t* orders_allowed, const char* error_msg);

int64_t CalcLeadingDimensionLt(int64_t rows, int64_t cols, cublasLtOrder_t order);

class CublasLtMMAlgoMap {
 public:
  static CublasLtMMAlgoMap& instance();

  void GetAlgo(cublasLtHandle_t cublasLt_handle, cublasLtMatmulAlgo_t& algo, const cudaDeviceProp& device_prop,
               int batch_count, int m, int n, int k,
               cublasLtOrder_t weight_order, cublasLtOrder_t input_output_order = CUBLASLT_ORDER_COL32) const;

  CublasLtMMAlgoMap(const CublasLtMMAlgoMap&) = delete;

  CublasLtMMAlgoMap& operator=(const CublasLtMMAlgoMap&) = delete;

 private:
  CublasLtMMAlgoMap();

  ~CublasLtMMAlgoMap() {}

 private:
  struct CublasLtMatmulAlgoInfo {
    int algoId, customOption, tile, splitK_val, swizzle, reductionScheme, workspaceSize, stages;
    float exec_time;
  };

  std::unordered_map<std::string, CublasLtMatmulAlgoInfo> best_algos_;
};

Status Reorder(cublasLtHandle_t cublasLt, cudaStream_t stream, const cudaDeviceProp& device_prop,
               int32_t batchCount, int64_t rows, int64_t cols, cudaDataType_t data_type,
               const void* input, cublasLtOrder_t order_input,
               void* output, cublasLtOrder_t order_output);

Status QOrdered_MatMul(cublasLtHandle_t cublasLt_handle, cudaStream_t stream, const cudaDeviceProp& device_prop,
                       int32_t batchCount, int64_t m, int64_t n, int64_t k,
                       const float* alpha,
                       const int8_t* A, const int8_t* B, bool isSingleBatchB,
                       const float* bias,
                       const float* beta,
                       const int8_t* C, bool isSingleBatchC,
                       int8_t* D,
                       cublasLtOrder_t order_weight);

inline Status QOrdered_MatMul(cublasLtHandle_t cublasLt_handle, cudaStream_t stream, const cudaDeviceProp& device_prop,
                              int32_t batchCount, int64_t m, int64_t n, int64_t k,
                              const float* alpha, const int8_t* A, const int8_t* B,
                              const float* bias, int8_t* C,
                              cublasLtOrder_t order_weight) {
  return QOrdered_MatMul(cublasLt_handle, stream, device_prop, batchCount, m, n, k,
                         alpha, A, B, true, bias, (const float*)nullptr, nullptr, false,
                         C, order_weight);
}

// using the following issue to locate error by reporting errors in the operator with error.
// #define DUBUG_PERF_CUDA_SYNC() CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize())
#define DUBUG_PERF_CUDA_SYNC()

// #endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime