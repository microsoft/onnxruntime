// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef USE_ROCM
#include "core/providers/rocm/shared_inc/rocm_utils.h"
#else
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#endif

namespace onnxruntime {
namespace cuda {

void FillOutputFromMaskImpl(cudaStream_t stream,
                            const int64_t total_element_count,
                            const BitmaskElementType* mask_data,
                            int* restored_output_mask);

void GetZeroPointRestoreTempStorageBytesImpl(cudaStream_t stream,
                                             size_t& temp_storage_bytes,
                                             int total_element_count);

void CalculateInputOffsetForEachOutputImpl(cudaStream_t stream,
                                           void* d_temp_storage,
                                           size_t& temp_storage_bytes,
                                           int* restored_output_mask,
                                           int* output_idx_to_input_idx_map_buffer,
                                           int total_element_count);

template <typename T>
void RestoreFromMaskImpl(const cudaDeviceProp& prop,
                         cudaStream_t stream,
                         const int64_t total_element_count,
                         const float zero_point_value,
                         const T* input_data,
                         const int* output_idx_to_input_idx_map_buffer,
                         T* output_data);

}  // namespace cuda
}  // namespace onnxruntime
