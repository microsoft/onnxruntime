// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace cuda {

template <typename T>
void GetTempStorageBytesImpl(cudaStream_t stream,
                             size_t& temp_storage_bytes,
                             T zero_point_value,
                             int total_element_count);

template <typename T>
void CopyOnConditionImpl(cudaStream_t stream,
                         void* d_temp_storage,
                         size_t& temp_storage_bytes,
                         const T* input_data,
                         T* output_buffer,
                         int& d_num_selected_out,
                         T zero_point_value,
                         int total_element_count);

template <typename T>
void SetMaskOutputImpl(const cudaDeviceProp& prop,
                       cudaStream_t stream,
                       const int64_t total_element_count,
                       const int64_t mask_element_count,
                       T zero_point_value,
                       const T* X_data,
                       void* mask_data);

}  // namespace cuda
}  // namespace onnxruntime
