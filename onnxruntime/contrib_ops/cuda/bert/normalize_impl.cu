// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "normalize_impl.h"

#include <algorithm>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, int blockSize>
__launch_bounds__(blockSize)
    __global__ void normalizeKernel(const T* input, T* output, int dim1, float* gamma, float* beta) {
  __shared__ float average_shared[blockSize];
  __shared__ float std_shared[blockSize];

  float average = 0.;
  int stride = dim1;
  const T* input_start = input + stride * blockIdx.x;
  T* output_start = output + stride * blockIdx.x;
  for (int i = threadIdx.x; i < dim1; i += blockSize) {
    average += (float)input_start[i];  // load from memory from native precision, convert to float for ops
  }
  average_shared[threadIdx.x] = average;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < blockSize; ++i) {
      average += average_shared[i];
    }
    average_shared[0] = (T)(average / dim1);  //typecast to native precision
  }
  __syncthreads();

  // std deviation
  average = average_shared[0];
  float stdDev = 0.;
  for (int i = threadIdx.x; i < dim1; i += blockSize) {
    float val = (float)(input_start[i]) - average;
    stdDev += val * val;
  }
  std_shared[threadIdx.x] = stdDev;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < blockSize; ++i) {
      stdDev += std_shared[i];
    }
    std_shared[0] = (T)(sqrtf(stdDev / dim1 + 1E-12));  //typecast to native precision
  }
  __syncthreads();
  stdDev = std_shared[0];

  for (int i = threadIdx.x; i < dim1; i += blockSize) {
    float x = input_start[i];
    output_start[i] = (T)(((x - average) / stdDev) * gamma[i] + beta[i]);
  }
}

template <typename T>

__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width, unsigned int mask = 0xffffffff)

{
#if CUDA_VERSION >= 9000
  return __shfl_sync(mask, value, srcLane, width);
#else
  return __shfl(value, srcLane, width);
#endif
}

template <typename T>

__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)

{
#if CUDA_VERSION >= 9000

  return __shfl_xor_sync(mask, value, laneMask, width);

#else

  return __shfl_xor(value, laneMask, width);

#endif
}

template <typename U>
__device__ void cuWelfordOnlineSum(
    const U curr,
    U& mu,
    U& sigma2,
    U& count) {
  count = count + U(1);
  U delta = curr - mu;
  U lmean = mu + delta / count;
  mu = lmean;
  U delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}

template <typename U>
__device__ void cuChanOnlineSum(
    const U muB,
    const U sigma2B,
    const U countB,
    U& mu,
    U& sigma2,
    U& count,
    const int& warp_size) {
  U delta = muB - mu;
  U nA = count;
  U nB = countB;
  count = count + countB;
  U nX = count;
  if (nX > U(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA * mu + nB * muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = U(0);
    sigma2 = U(0);
  }
}

template <typename T, typename U>
__device__ void cuWelfordMuSigma2(
    const T* __restrict__ vals,
    const int n1,
    const int n2,
    const int i1,
    U& mu,
    U& sigma2,
    U* buf,
    const int warp_size) {
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  U count = U(0);
  mu = U(0);
  sigma2 = U(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const T* lvals = vals + i1 * n2;
    int l = 4 * thrx;
    for (; l + 3 < n2; l += 4 * numx) {
      for (int k = 0; k < 4; ++k) {
        U curr = static_cast<U>(lvals[l + k]);
        cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
      }
    }
    for (; l < n2; ++l) {
      U curr = static_cast<U>(lvals[l]);
      cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
    }
    // intra-warp reductions
    for (int l = 0; l <= 4; ++l) {
      int srcLaneB = (threadIdx.x + (1 << l)) & 31;
      U muB = WARP_SHFL(mu, srcLaneB, warp_size);
      U countB = WARP_SHFL(count, srcLaneB, warp_size);
      U sigma2B = WARP_SHFL(sigma2, srcLaneB, warp_size);
      cuChanOnlineSum<U>(muB, sigma2B, countB, mu, sigma2, count, warp_size);
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      U* ubuf = (U*)buf;
      U* ibuf = (U*)(ubuf + blockDim.y);
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2 * wrt_y] = mu;
          ubuf[2 * wrt_y + 1] = sigma2;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          U muB = ubuf[2 * threadIdx.y];
          U sigma2B = ubuf[2 * threadIdx.y + 1];
          U countB = ibuf[threadIdx.y];
          cuChanOnlineSum<U>(muB, sigma2B, countB, mu, sigma2, count, warp_size);
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ubuf[0] = mu;
        ubuf[1] = sigma2;
      }
      __syncthreads();
      mu = ubuf[0];
      sigma2 = ubuf[1] / U(n2);
      // don't care about final value of count, we know count == n2
    } else {
      mu = WARP_SHFL(mu, 0, warp_size);
      sigma2 = WARP_SHFL(sigma2 / U(n2), 0, warp_size);
    }
  }
}

template <>
__device__ void cuWelfordMuSigma2(
    const half* __restrict__ vals,
    const int n1,
    const int n2,
    const int i1,
    float& mu,
    float& sigma2,
    float* buf,
    const int warp_size) {
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  float count = 0.0f;
  mu = float(0);
  sigma2 = float(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const half* lvals = vals + i1 * n2;
    int l = 8 * thrx;
    if ((((size_t)lvals) & 3) != 0) {
      // 16 bit alignment
      // first thread consumes first point
      if (thrx == 0) {
        float curr = static_cast<float>(lvals[0]);
        cuWelfordOnlineSum(curr, mu, sigma2, count);
      }
      ++l;
    }
    // at this point, lvals[l] are 32 bit aligned for all threads.
    for (; l + 7 < n2; l += 8 * numx) {
      for (int k = 0; k < 8; k += 2) {
        float2 curr = __half22float2(*((__half2*)(lvals + l + k)));
        cuWelfordOnlineSum(curr.x, mu, sigma2, count);
        cuWelfordOnlineSum(curr.y, mu, sigma2, count);
      }
    }
    for (; l < n2; ++l) {
      float curr = static_cast<float>(lvals[l]);
      cuWelfordOnlineSum(curr, mu, sigma2, count);
    }
    // intra-warp reductions
    for (int l = 0; l <= 4; ++l) {
      int srcLaneB = (threadIdx.x + (1 << l)) & 31;
      float muB = WARP_SHFL(mu, srcLaneB, warp_size);
      float countB = WARP_SHFL(count, srcLaneB, warp_size);
      float sigma2B = WARP_SHFL(sigma2, srcLaneB, warp_size);
      cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count, warp_size);
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float* ubuf = (float*)buf;
      float* ibuf = (float*)(ubuf + blockDim.y);
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2 * wrt_y] = mu;
          ubuf[2 * wrt_y + 1] = sigma2;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          float muB = ubuf[2 * threadIdx.y];
          float sigma2B = ubuf[2 * threadIdx.y + 1];
          float countB = ibuf[threadIdx.y];
          cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count, warp_size);
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ubuf[0] = mu;
        ubuf[1] = sigma2;
      }
      __syncthreads();
      mu = ubuf[0];
      sigma2 = ubuf[1] / float(n2);
      // don't care about final value of count, we know count == n2
    } else {
      mu = WARP_SHFL(mu, 0, warp_size);
      sigma2 = WARP_SHFL(sigma2 / float(n2), 0, warp_size);
    }
  }
}

template <typename U>
__device__ U rsqrt(U v) {
  return U(1) / sqrt(v);
}
template <>
__device__ float rsqrt(float v) {
  return rsqrtf(v);
}
template <>
__device__ double rsqrt(double v) {
  return rsqrt(v);
}

namespace {
// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
//  template <typename T>
//  struct SharedMemory
//  {
//      // Ensure that we won't compile any un-specialized types
//      __device__ T *getPointer()
//      {
//          extern __device__ void error(void);
//          error();
//          return NULL;
//      }
//  };
// https://github.com/NVIDIA/apex/issues/246
template <typename T>
struct SharedMemory;

template <>
struct SharedMemory<float> {
  __device__ float* getPointer() {
    extern __shared__ float s_float[];
    return s_float;
  }
};

template <>
struct SharedMemory<double> {
  __device__ double* getPointer() {
    extern __shared__ double s_double[];
    return s_double;
  }
};
}  // namespace

template <typename T, typename U>
__global__ void cuApplyLayerNorm(
    T* __restrict__ output_vals,
    U* __restrict__ mean,
    U* __restrict__ invvar,
    const T* __restrict__ vals,
    const int n1,
    const int n2,
    const U epsilon,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    int warp_size) {
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensors are contiguous
  //
  for (auto i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    U mu, sigma2;
    cuWelfordMuSigma2(vals, n1, n2, i1, mu, sigma2, buf, warp_size);
    const T* lvals = vals + i1 * n2;
    T* ovals = output_vals + i1 * n2;
    U c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL && beta != NULL) {
      for (int i = thrx; i < n2; i += numx) {
        U curr = static_cast<U>(lvals[i]);
        ovals[i] = gamma[i] * static_cast<T>(c_invvar * (curr - mu)) + beta[i];
      }
    } else {
      for (int i = thrx; i < n2; i += numx) {
        U curr = static_cast<U>(lvals[i]);
        ovals[i] = static_cast<T>(c_invvar * (curr - mu));
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0 && mean != nullptr && invvar != nullptr) {
      mean[i1] = mu;
      invvar[i1] = c_invvar;
    }
  }
}

template <typename T, typename U>
void HostApplyLayerNorm(
    T* output,
    U* mean,
    U* invvar,
    const T* input,
    int64_t n1,
    int64_t n2,
    const T* gamma,
    const T* beta,
    double epsilon = 1e-12) {
  const dim3 threads(32, 4, 1);
  const cudaDeviceProp& prop = GridDim::GetDeviceProps();
  const uint64_t maxGridY = prop.maxGridSize[1];
  const int warp_size = prop.warpSize;
  //  const uint64_t maxGridY = 32;
  const dim3 blocks(1, std::min((uint64_t)n1, maxGridY), 1);
  int nshared =
      threads.y > 1 ? threads.y * sizeof(U) + (threads.y / 2) * sizeof(U) : 0;
  cuApplyLayerNorm<<<blocks, threads, nshared, 0>>>(
      output,
      mean,
      invvar,
      input,
      n1, n2,
      U(epsilon),
      gamma, beta, warp_size);
}

#define LAYERNORM_LINEAR_IMPL(T, U)                                                                       \
  template void HostApplyLayerNorm(T* output, U* mean, U* invvar, const T* input, int64_t n1, int64_t n2, \
                                   const T* gamma, const T* beta, double epsilon = 1e-12);

LAYERNORM_LINEAR_IMPL(float, float)
LAYERNORM_LINEAR_IMPL(half, float)
LAYERNORM_LINEAR_IMPL(double, float)
//LAYERNORM_LINEAR_IMPL(half, half)

void launchNormalizeKernel(const float* input,
                           float* output,
                           float* gamma_ptr,  // gamma
                           float* beta_ptr,   // beta
                           int nBatch,
                           int sequence_len,
                           int encode_len  //,
                           /*int isFP16*/) {
  // size_t elementSize = isFP16 ? 2 : 4;
  //const int blockSize = 32;
  //const int gridSize = sequence_len * nBatch;
  //
  ////if( isFP16 )
  ////   normalizeKernel<__half, blockSize> << <gridSize, blockSize, 0, stream >> > ( static_cast< const __half* >( input ),
  ////      static_cast< __half* >( output ), meanArray, stdArray, d[ 1 ], params[ 0 ], params[ 1 ] );
  ////else
  //normalizeKernel<float, blockSize> << <gridSize, blockSize, 0 >> > ( input, output, encode_len, gamma_ptr, beta_ptr );

  HostApplyLayerNorm<float, float>(
      output,
      nullptr,
      nullptr,
      input,
      sequence_len * nBatch,
      encode_len,
      gamma_ptr,
      beta_ptr);
}

}  // namespace cuda
}  //namespace contrib
}  // namespace onnxruntime
