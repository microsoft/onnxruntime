// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/tensor/onehot.h"

namespace onnxruntime {
namespace cuda {

template <typename in_type, typename out_type>
__global__ void _OneHotImpl(
    const in_type* indices_data,
    const fast_divmod fdm_depth_suffix,
    const fast_divmod fdm_suffix,
    const int64_t depth_val,
    const out_type on_value,
    const out_type off_value,
    out_type* output_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  int prefix_index, prefix_offset;
  fdm_depth_suffix.divmod(id, prefix_index, prefix_offset);

  int depth_index, suffix_index;
  fdm_suffix.divmod(prefix_offset, depth_index, suffix_index);

  CUDA_LONG indices_index = prefix_index * fdm_suffix.d_ + suffix_index;

  // handle index outside the range [-depth, depth-1] case
  bool is_valid_range = indices_data[indices_index] >= -depth_val && indices_data[indices_index] < depth_val;

  // handle negative index
  in_type adjusted_indice = (indices_data[indices_index] + depth_val) % depth_val;

  output_data[id] = (is_valid_range && adjusted_indice == in_type(depth_index)) ? on_value : off_value;
}

template<typename in_type, typename out_type>
__global__ void _OneHotWithZeroOffValueImpl(
    const in_type* indices_data,
    const fast_divmod fdm_suffix,
    const int64_t depth_val,
    const out_type on_value,
    out_type* output_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  if (indices_data[id] >= -depth_val && indices_data[id] < depth_val) {
    in_type adjusted_index = indices_data[id] >= 0 ? indices_data[id] : indices_data[id] + depth_val;
    int q, r;
    fdm_suffix.divmod(id, q, r);
    output_data[(q * depth_val + adjusted_index) * fdm_suffix.d_ + r] = on_value;
  }
}

template <typename in_type, typename out_type>
void OneHotImpl(
    cudaStream_t stream,
    const in_type* indices_data,
    const fast_divmod fdm_depth_suffix,
    const fast_divmod fdm_suffix,
    const int64_t depth_val,
    const out_type on_value,
    const out_type off_value,
    out_type* output_data,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _OneHotImpl<in_type, out_type><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
    indices_data,
    fdm_depth_suffix,
    fdm_suffix,
    depth_val,
    on_value,
    off_value,
    output_data,
    N);
}

template <typename in_type, typename out_type>
void OneHotWithZeroOffValueImpl(
    cudaStream_t stream,
    const in_type* indices_data,
    const fast_divmod fdm_suffix,
    const int64_t depth_val,
    const out_type on_value,
    out_type* output_data,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _OneHotWithZeroOffValueImpl<in_type, out_type><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
    indices_data,
    fdm_suffix,
    depth_val,
    on_value,
    output_data,
    N);
}

#define SPECIALIZED_OneHotImpl(in_type, out_type) \
  template void OneHotImpl(                       \
    cudaStream_t stream,                          \
    const in_type* indices_data,                  \
    const fast_divmod fdm_depth_suffix,           \
    const fast_divmod fdm_suffix,                 \
    const int64_t depth_val,                      \
    const out_type on_value,                      \
    const out_type off_value,                     \
    out_type* output_data,                        \
    size_t count);

SPECIALIZED_OneHotImpl(int64_t, int64_t)
SPECIALIZED_OneHotImpl(int64_t, float)
SPECIALIZED_OneHotImpl(int32_t, float)
SPECIALIZED_OneHotImpl(int64_t, half)
SPECIALIZED_OneHotImpl(int32_t, half)

#define SPECIALIZED_OneHotWithZeroOffValueImpl(in_type, out_type) \
  template void OneHotWithZeroOffValueImpl(                       \
    cudaStream_t stream,                                          \
    const in_type* indices_data,                                  \
    const fast_divmod fdm_suffix,                                 \
    const int64_t depth_val,                                      \
    const out_type on_value,                                      \
    out_type* output_data,                                        \
    size_t count);

SPECIALIZED_OneHotWithZeroOffValueImpl(int64_t, int64_t)
SPECIALIZED_OneHotWithZeroOffValueImpl(int64_t, float)
SPECIALIZED_OneHotWithZeroOffValueImpl(int32_t, float)
SPECIALIZED_OneHotWithZeroOffValueImpl(int64_t, half)
SPECIALIZED_OneHotWithZeroOffValueImpl(int32_t, half)

}  // namespace cuda
}  // namespace onnxruntime
