// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/nn/conv_impl.h"

#include "core/providers/rocm/cu_inc/binary_elementwise_impl.cuh"
#include "core/providers/rocm/math/binary_elementwise_ops_impl_functors.cuh"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fast_divmod.h"

namespace onnxruntime {
namespace rocm {

template <typename T, typename T1, typename T2>
void ConvBiasImpl(
    hipStream_t stream,
    const T1* lhs_data,
    const T2* rhs_data,
    T* output_data,
    size_t bias_size,
    size_t count) {
  int output_rank_or_simple_broadcast = static_cast<int32_t>(SimpleBroadcast::RightPerChannelBatchN);
  fast_divmod fdm_h(1);
  fast_divmod fdm_c(bias_size);

  BinaryElementWiseImpl(stream, output_rank_or_simple_broadcast, nullptr, lhs_data, nullptr, rhs_data,
                        nullptr, fdm_h, fdm_c, output_data, OP_Add<T, T1, T2>(),
                        count);
}

template void ConvBiasImpl<float, float, float>(
    hipStream_t stream,
    const float* lhs_data,
    const float* rhs_data,
    float* output_data,
    size_t bias_size,
    size_t count);

template void ConvBiasImpl<half, half, half>(
    hipStream_t stream,
    const half* lhs_data,
    const half* rhs_data,
    half* output_data,
    size_t bias_size,
    size_t count);

}  // namespace rocm
}  // namespace onnxruntime
