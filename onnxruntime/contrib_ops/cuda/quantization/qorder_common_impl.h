
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
void QOrderQuantize(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const T* src, int8_t* dst, float scale, size_t N);

// internally using fp32 computation to avoid precision lost
void QOrderQuantize_Strict(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const __half* src, int8_t* dst, float scale, size_t N);

template <typename T>
void QOrderDequantize(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, T* dst, float scale, size_t N);

void QOrderDequantize_Strict(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, __half* dst, float scale, size_t N);

// template <typename T>
// void QOrderDequantizeToRow(
//     cublasLtOrder_t input_order,
//     cudaStream_t stream, const cudaDeviceProp& device_prop,
//     const int8_t* src, T* dst, float scale, unsigned batch, unsigned rows, unsigned cols);

// template <typename T>
// void QOrderQuantizeRowTo(
//     cublasLtOrder_t output_order,
//     cudaStream_t stream, const cudaDeviceProp& device_prop,
//     const T* src, int8_t* dst, float scale, unsigned batch, unsigned rows, unsigned cols);

void QOrderDequantizeToRow(
    cublasLtOrder_t input_order, cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, __half* dst, float scale, unsigned batch, unsigned rows, unsigned cols);

void QOrderQuantizeRowTo(
    cublasLtOrder_t output_order, cudaStream_t stream, const cudaDeviceProp& device_prop,
    const __half* src, int8_t* dst, float scale, unsigned batch, unsigned rows, unsigned cols);

void QOrderQuantizeRowToCol32(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const __half* src, int8_t* dst, float scale, unsigned batch, unsigned rows, unsigned cols);

void QOrderQuantizeRowToCol32(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const float* src, int8_t* dst, float scale, unsigned batch, unsigned rows, unsigned cols);

void QOrderDequantizeCol32ToRow(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, __half* dst, float scale, unsigned batch, unsigned rows, unsigned cols);

void QOrderDequantizeCol32ToRow(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, float* dst, float scale, unsigned batch, unsigned rows, unsigned cols);

template <typename T>
void QOrderLayerNorm(
    cudaStream_t stream, const cudaDeviceProp& device_prop, cublasLtOrder_t order,
    const int8_t* src, const float src_scale, int8_t* dst, const float dst_scale,
    const T* gamma, const T* beta, const float epsilon,
    unsigned batch, unsigned rows, unsigned cols);

void ReorderS8RowToCol32(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, int8_t* dst,
    unsigned batch, unsigned rows, unsigned cols);

// mask_index is (batch, sequence_len)
void QOrderMaskedSoftmax(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, const float* lookup_table,
    const int32_t* mask_index,
    int8_t* dst, const float scale_dst,
    const unsigned batch, const unsigned num_heads, const unsigned sequence_len);

void BuildTableForSoftmaxPowerOf(cudaStream_t stream, const float base, float* table);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
