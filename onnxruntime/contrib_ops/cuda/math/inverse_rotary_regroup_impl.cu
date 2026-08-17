// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/inverse_rotary_regroup_impl.h"

#include "contrib_ops/cuda/math/quant_sim_common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int kThreads = 256;

// Undo the rotation applied to the query and lay the heads out by output-projection group.
//
// The attention output arrives as [tokens, num_heads * head_dim]; the grouped projection wants
// [num_groups, tokens, group_dim]. Both views index the same flat channel f = h * head_dim + c,
// so the reshape/transpose/reshape trio the graph uses is just an addressing change here.
template <typename CudaT>
__global__ void InverseRotaryRegroupKernel(const InverseRotaryRegroupParams p,
                                           const CudaT* __restrict__ input,
                                           const float* __restrict__ cos_table,
                                           const float* __restrict__ sin_table,
                                           CudaT* __restrict__ output) {
  const int64_t total = static_cast<int64_t>(p.num_tokens) * p.num_heads * p.head_dim;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += stride) {
    const int c = static_cast<int>(idx % p.head_dim);
    const int64_t row = idx / p.head_dim;
    const int h = static_cast<int>(row % p.num_heads);
    const int64_t token = row / p.num_heads;

    float v = QuantSimConv<CudaT>::ToFloat(input[idx]);
    if (c >= p.nope_dim) {
      // nope_dim is even, so the pair partner of channel c is idx ^ 1 in the flat layout too.
      const int t = c - p.nope_dim;
      const float partner = QuantSimConv<CudaT>::ToFloat(input[idx ^ 1]);
      // Inverse of the forward rotation: R_inv = -R, so the signed swap flips sign.
      const float rot = (t & 1) ? -partner : partner;
      const int64_t base = token * p.rope_head_dim + t;
      v = v * cos_table[base] + rot * sin_table[base];
    }

    const int f = h * p.head_dim + c;
    const int g = f / p.group_dim;
    const int inner = f - g * p.group_dim;
    const int64_t out_idx =
        (static_cast<int64_t>(g) * p.num_tokens + token) * p.group_dim + inner;
    output[out_idx] = QuantSimConv<CudaT>::FromFloat(v);
  }
}

}  // namespace

template <typename T>
Status LaunchInverseRotaryRegroup(cudaStream_t stream, const InverseRotaryRegroupParams& p,
                                  const T* input, const float* cos_table, const float* sin_table,
                                  T* output) {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const int64_t total = static_cast<int64_t>(p.num_tokens) * p.num_heads * p.head_dim;
  if (total == 0) return Status::OK();

  const int64_t want = (total + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(want < 65535 ? want : 65535);
  InverseRotaryRegroupKernel<CudaT><<<blocks, kThreads, 0, stream>>>(
      p, reinterpret_cast<const CudaT*>(input), cos_table, sin_table,
      reinterpret_cast<CudaT*>(output));
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

#define INSTANTIATE(T)                                                                           \
  template Status LaunchInverseRotaryRegroup<T>(cudaStream_t, const InverseRotaryRegroupParams&, \
                                                const T*, const float*, const float*, T*)

INSTANTIATE(float);
INSTANTIATE(MLFloat16);
INSTANTIATE(BFloat16);

#undef INSTANTIATE

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
