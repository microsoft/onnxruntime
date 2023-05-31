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
