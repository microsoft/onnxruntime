// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "quantize_linear.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <class T, class U>
Status CudaQuantizeLinearStd(cudaStream_t stream, const U* input, T* output, const U* scale, const T* zero_point, size_t num_of_element);

template <class T, class U>
Status CudaQuantizeLinearSat(cudaStream_t stream, const U* input, T* output, const U* scale, const T* zero_point, size_t num_of_element,
                             bool saturate);

template <class T, class U>
Status CudaQuantizeLinearAxisStd(cudaStream_t stream, const U* input, T* output, const U* scale, const T* zero_point, size_t num_of_element,
                                 size_t batch_size, size_t n_scales);

template <class T, class U>
Status CudaQuantizeLinearAxisSat(cudaStream_t stream, const U* input, T* output, const U* scale, const T* zero_point, size_t num_of_element,
                                 size_t batch_size, size_t n_scales, bool saturate);

/**
 * @brief block-wise quantization with standard rounding to int4. Input is reshaped to [M, K, N]. K the quantization
 *        axis. Scale is reshaped to [M, ceil(K/block_size), N]. For an index i in input, the coordiate is (xi, yi, zi)
 *        = (i / (K * N), i % (K * N) / N, i % N). The scale coordiate is (xi, yi / block_size, zi). The scale index
 *        is xi * ceil(K / block_size) * N + yi / block_size * N + zi. The zero point is reshaped to [M, K, N].
 * @tparam OutT           quantized type, int8_t for Int4x2, uint8_t for UInt4x2
 * @tparam InT            full precision type
 * @param stream          cuda stream
 * @param input           input tensor
 * @param output          output tensor
 * @param scale           scale tensor
 * @param zero_point      zero point tensor
 * @param num_of_element  number of elements in input tensor
 * @param K               K
 * @param N               N
 * @param block_size      block size
 */
template <class OutT, class InT>
Status CudaQuantizeLinearBlockStdInt4(cudaStream_t stream, const InT* input, OutT* output, const InT* scale,
                                      const OutT* zero_point, size_t num_of_element, size_t K, size_t N,
                                      size_t block_size);

template <class T, class U>
Status CudaDequantizeLinearStd(cudaStream_t stream, const T* input, U* output, const U* scale, const T* zero_point, size_t num_of_element);

template <class T, class U>
Status CudaDequantizeLinearSat(cudaStream_t stream, const T* input, U* output, const U* scale, const T* zero_point, size_t num_of_element);

template <class T, class U>
Status CudaDequantizeLinearAxisStd(cudaStream_t stream, const T* input, U* output, const U* scale, const T* zero_point, size_t num_of_element,
                                   size_t batch_size, size_t n_scales);

template <class T, class U>
Status CudaDequantizeLinearAxisSat(cudaStream_t stream, const T* input, U* output, const U* scale, const T* zero_point, size_t num_of_element,
                                   size_t batch_size, size_t n_scales);

}  // namespace cuda
}  // namespace onnxruntime
