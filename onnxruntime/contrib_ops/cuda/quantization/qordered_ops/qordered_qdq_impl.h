// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11040

template <typename T>
Status QOrderQuantize(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const T* src, int8_t* dst, float scale, size_t N);

// internally using fp32 computation to avoid precision lost
Status QOrderQuantize_Strict(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const __half* src, int8_t* dst, float scale, size_t N);

template <typename T>
Status QOrderDequantize(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, T* dst, float scale, size_t N);

Status QOrderDequantize_Strict(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, __half* dst, float scale, size_t N);

Status QOrderDequantizeToRow(
    cublasLtOrder_t input_order, cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, __half* dst, float scale, unsigned batch, unsigned rows, unsigned cols);

Status QOrderQuantizeRowTo(
    cublasLtOrder_t output_order, cudaStream_t stream, const cudaDeviceProp& device_prop,
    const __half* src, int8_t* dst, float scale, unsigned batch, unsigned rows, unsigned cols);

Status QOrderQuantizeRowToCol32(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const __half* src, int8_t* dst, float scale, unsigned batch, unsigned rows, unsigned cols);

Status QOrderQuantizeRowToCol32(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const float* src, int8_t* dst, float scale, unsigned batch, unsigned rows, unsigned cols);

Status QOrderDequantizeCol32ToRow(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, __half* dst, float scale, unsigned batch, unsigned rows, unsigned cols);

Status QOrderDequantizeCol32ToRow(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, float* dst, float scale, unsigned batch, unsigned rows, unsigned cols);

Status Reorder(
    cublasLtHandle_t cublasLt, cudaStream_t stream, const cudaDeviceProp& device_prop,
    int32_t batchCount, int64_t rows, int64_t cols, cudaDataType_t data_type,
    const void* input, cublasLtOrder_t order_input,
    void* output, cublasLtOrder_t order_output);

Status ReorderS8RowToCol32(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, int8_t* dst,
    unsigned batch, unsigned rows, unsigned cols);

Status CheckTensorOrder(
    const Tensor& input_tensor, cublasLtOrder_t input_order, cublasLtOrder_t output_order,
    int64_t& rows, int64_t& cols, int64_t& batchCount, int64_t& elementCount);

cublasLtOrder_t GetCublasLtOrderAttr(
    const OpKernelInfo& info, const char* order_attr);

cublasLtOrder_t GetCublasLtOrderAttr(
    const OpKernelInfo& info, const char* order_attr,
    int num_allowed_orders, const cublasLtOrder_t* orders_allowed, const char* error_msg);

int64_t CalcLeadingDimensionLt(
    int64_t rows, int64_t cols, cublasLtOrder_t order);

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
