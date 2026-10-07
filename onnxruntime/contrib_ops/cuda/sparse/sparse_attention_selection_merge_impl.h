#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace onnxruntime::contrib::cuda {

cudaError_t LaunchSparseAttentionSelectionMerge(
    cudaStream_t stream, const int32_t* base_indices, const int32_t* base_counts,
    const int32_t* base_rows, const int32_t* starts, const int32_t* ends,
    int32_t* indices, int32_t* counts, int32_t* status, int32_t* workspace,
    int32_t cached_rows, int32_t base_capacity, int32_t query_count,
    int32_t output_capacity, int32_t hash_capacity);

}