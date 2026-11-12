// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <math_constants.h>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/cub.cuh"
#include "core/providers/cuda/math/logcumsumexp_impl.h"
#include "core/providers/cuda/shared_inc/accumulation_type.h"

namespace onnxruntime {
namespace cuda {
namespace {

constexpr int kBlockSize = 256;

template <typename T>
struct LogAddExp {
  __device__ T operator()(T a, T b) const {
    // NaN must dominate even positive infinity, independent of scan order.
    if (isnan(a)) return a;
    if (isnan(b)) return b;
    const T maximum = a > b ? a : b;
    if (isinf(maximum)) return maximum;
    const T minimum = a < b ? a : b;
    return maximum + log1p(exp(minimum - maximum));
  }
};

template <typename T>
__global__ void LogCumSumExpBlockScan(const T* input, T* output, int64_t width, int64_t inner, int64_t lanes) {
  using A = AccumulationType_t<T>;
  using Scan = cub::BlockScan<A, kBlockSize>;
  __shared__ typename Scan::TempStorage storage;
  __shared__ A carry;
  const A negative_infinity = -static_cast<A>(CUDART_INF);
  const LogAddExp<A> combine{};

  for (int64_t lane = blockIdx.x; lane < lanes; lane += gridDim.x) {
    const int64_t base = (lane / inner) * width * inner + lane % inner;
    if (threadIdx.x == 0) carry = negative_infinity;
    __syncthreads();
    for (int64_t tile = 0; tile < width; tile += kBlockSize) {
      const int64_t j = tile + threadIdx.x;
      A value = j < width ? static_cast<A>(input[base + j * inner]) : negative_infinity;
      // Seed the first thread so all prefixes include the preceding tiles.
      if (threadIdx.x == 0) value = combine(carry, value);
      A prefix, aggregate;
      Scan(storage).InclusiveScan(value, prefix, combine, aggregate);
      if (j < width) output[base + j * inner] = static_cast<T>(prefix);
      __syncthreads();
      if (threadIdx.x == 0) carry = aggregate;
      __syncthreads();
    }
  }
}

template <typename T>
__global__ void LogCumSumExpStrided(const T* input, T* output, int64_t width, int64_t inner, int64_t lanes) {
  using A = AccumulationType_t<T>;
  const LogAddExp<A> combine{};
  // Neighboring threads scan neighboring inner indices for coalesced accesses.
  for (int64_t lane = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       lane < lanes; lane += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t offset = (lane / inner) * width * inner + lane % inner;
    A prefix = -static_cast<A>(CUDART_INF);
    for (int64_t j = 0; j < width; ++j, offset += inner) {
      prefix = combine(prefix, static_cast<A>(input[offset]));
      output[offset] = static_cast<T>(prefix);
    }
  }
}

}  // namespace

template <typename T>
Status LogCumSumExpImpl(cudaStream_t stream, const T* input, T* output,
                        int64_t width, int64_t inner, int64_t lanes) {
  // With few independent lanes, cooperate along the scan axis instead of
  // leaving nearly all threads idle on a long strided scan.
  if (inner == 1 || lanes <= kBlockSize) {
    const int blocks = static_cast<int>(std::min<int64_t>(lanes, 65535));
    LogCumSumExpBlockScan<<<blocks, kBlockSize, 0, stream>>>(input, output, width, inner, lanes);
  } else {
    const int blocks = static_cast<int>(std::min<int64_t>((lanes - 1) / kBlockSize + 1, 65535));
    LogCumSumExpStrided<<<blocks, kBlockSize, 0, stream>>>(input, output, width, inner, lanes);
  }
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE(T)                                                                \
  template Status LogCumSumExpImpl<T>(cudaStream_t stream, const T* input, T* output, \
                                      int64_t width, int64_t inner, int64_t lanes);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(half)
INSTANTIATE(BFloat16)
#undef INSTANTIATE

}  // namespace cuda
}  // namespace onnxruntime
