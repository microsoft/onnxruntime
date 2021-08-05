/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include <cuda_fp16.h>
#include <thrust/tuple.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/accumulation_type.h"
#include "layer_norm_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

constexpr int kCUDANumThreads = 256;
//constexpr int kColwiseReduceTileSize = 32;

#if defined(__CUDACC__) || defined(__HIPCC__)
#define C10_HOST_DEVICE __host__ __device__
#define C10_DEVICE __device__
#define C10_HOST __host__
#else
#define C10_HOST_DEVICE
#define C10_HOST
#define C10_DEVICE
#endif

#ifdef __HIP_PLATFORM_HCC__
#define C10_WARP_SIZE 64
#else
#define C10_WARP_SIZE 32
#endif

#if defined(__clang__)
  #define __ubsan_ignore_float_divide_by_zero__ __attribute__((no_sanitize("float-divide-by-zero")))
  //#define __ubsan_ignore_undefined__ __attribute__((no_sanitize("undefined")))
  //#define __ubsan_ignore_signed_int_overflow__ __attribute__((no_sanitize("signed-integer-overflow")))
#else
  #define __ubsan_ignore_float_divide_by_zero__
  //#define __ubsan_ignore_undefined__
  //#define __ubsan_ignore_signed_int_overflow__
#endif

#if defined(__HIP_PLATFORM_HCC__)
// take these out when ROCm implements std:: math functions
#include <math.h>
template <typename scalar_t>
static __forceinline__ __device__ scalar_t device_sqrt(scalar_t val);

template <>
__forceinline__ __device__ float device_sqrt(float val) {
  return ::sqrtf(val);
}

template <>
__forceinline__ __device__ double device_sqrt(double val) {
  return ::sqrt(val);
}
#else
template<typename scalar_t>
__forceinline__ __device__ double device_sqrt(scalar_t val) {
  return std::sqrt(val);
}
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/pair.h>
#else
#include <cmath>
#define device_sqrt std::sqrt
#endif

/*
template <typename T, bool is_cuda>
struct AccumulateType { };

#if defined(__CUDACC__) || defined(__HIPCC__)
template <> struct AccumulateType<half, true> { using type = float; };
#endif

#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
template <> struct AccumulateType<nv_bfloat16, true> {using type = float; };
#endif

template <> struct AccumulateType<float, true> { using type = float; };
template <> struct AccumulateType<double, true> { using type = double; };
template <> struct AccumulateType<int8_t, true> { using type = int64_t; };
template <> struct AccumulateType<uint8_t, true> { using type = int64_t; };
template <> struct AccumulateType<char, true> { using type = int64_t; };
template <> struct AccumulateType<int16_t, true> { using type = int64_t; };
template <> struct AccumulateType<int32_t, true> { using type = int64_t; };
template <> struct AccumulateType<int64_t, true> { using type = int64_t; };
template <> struct AccumulateType<bool, true> {using type = bool; };

template <> struct AccumulateType<float, false> { using type = double; };
template <> struct AccumulateType<double, false> { using type = double; };
template <> struct AccumulateType<int8_t, false> { using type = int64_t; };
template <> struct AccumulateType<uint8_t, false> { using type = int64_t; };
template <> struct AccumulateType<char, false> { using type = int64_t; };
template <> struct AccumulateType<int16_t, false> { using type = int64_t; };
template <> struct AccumulateType<int32_t, false> { using type = int64_t; };
template <> struct AccumulateType<int64_t, false> { using type = int64_t; };
template <> struct AccumulateType<bool, false> {using type = bool; };

template<typename T, bool is_cuda>
using acc_type = typename AccumulateType<T, is_cuda>::type;
*/


constexpr int kCUDABlockReduceNumThreads = 512;

// Sums `val` accross all threads in a warp.
//
// Assumptions:
//   - The size of each block should be a multiple of `C10_WARP_SIZE`
template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

// Sums `val` accross all threads in a block.
//
// Assumptions:
//   - Thread blocks are an 1D set of threads (indexed with `threadIdx.x` only)
//   - The size of each block should be a multiple of `C10_WARP_SIZE`
//   - `shared` should be a pointer to shared memory with size of, at least,
//     `sizeof(T) * number_of_warps`
template <typename T>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  val = WarpReduceSum(val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < blockDim.x / C10_WARP_SIZE) ? shared[lid] : 0;
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

template <typename T, class ReduceOp>
__inline__ __device__ T WarpReduce(T val, const ReduceOp& op) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val = op.combine(val, op.warp_shfl_down(val, offset));
  }
  return val;
}

template <typename T, class ReduceOp>
__inline__ __device__ T
BlockReduce(T val, const ReduceOp& op, const T& identity_element, T* shared) {
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  val = WarpReduce(val, op);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < blockDim.x / C10_WARP_SIZE) ? shared[lid]
                                                   : identity_element;
  if (wid == 0) {
    val = WarpReduce(val, op);
  }
  return val;
}


template <typename scalar_t, typename index_t, typename combine_t>
struct WelfordData {
  scalar_t mean;
  scalar_t m2;
  index_t n;
  combine_t nf;

  C10_HOST_DEVICE WelfordData() : mean(0), m2(0), n(0), nf(0) {}

  C10_HOST_DEVICE WelfordData(
      scalar_t mean,
      scalar_t m2,
      index_t n,
      combine_t nf)
      : mean(mean), m2(m2), n(n), nf(nf) {}
};

template <typename scalar_t, typename acc_scalar_t, typename index_t, typename combine_t, typename res_t>
struct WelfordOps {
  index_t correction;
  bool take_sqrt;
 public:
  using acc_t = WelfordData<acc_scalar_t, index_t, combine_t>;
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, index_t /*idx*/) const {
    acc_scalar_t delta = data - acc.mean;
    // using acc.nf(combine_t) here, as acc.n(index_t) would still be converted
    // accumulation in reduce is done through index_T
    acc_scalar_t new_mean = acc.mean + delta / (acc.nf + 1);
    acc_scalar_t new_delta = data - new_mean;
    return {
      new_mean,
      acc.m2 + delta * new_delta,
      acc.n + 1,
      combine_t(acc.n + 1), // accumulate for combine_t uses index_t
    };
  }
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    if (a.nf == 0) {
      return b;
    }
    if (b.nf == 0) {
      return a;
    }
    acc_scalar_t delta = b.mean - a.mean;
    combine_t new_count = a.nf + b.nf;
    acc_scalar_t nb_over_n = b.nf / new_count;
    return {
      a.mean + delta * nb_over_n,
      a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
      // setting acc.n as -1 since acc.n might not be able to represent the count
      // correctly within its range, setting it to -1 to avoid confusion
      -1,
      new_count
    };
  }

  inline C10_DEVICE res_t project(acc_t acc) const __ubsan_ignore_float_divide_by_zero__ {
    const auto mean = static_cast<scalar_t>(acc.mean);
    const combine_t divisor = acc.nf > correction ? acc.nf - correction : 0;
    const auto var = acc.m2 / divisor;
    res_t results(take_sqrt ? device_sqrt(var) : var, mean);
    return results;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline __device__ acc_t warp_shfl_down(acc_t acc, int offset) const {
    return {
      WARP_SHFL_DOWN(acc.mean, offset)
      , WARP_SHFL_DOWN(acc.m2, offset)
      , WARP_SHFL_DOWN(acc.n, offset)
      , WARP_SHFL_DOWN(acc.nf, offset)
    };
  }
#endif
  C10_HOST_DEVICE WelfordOps(index_t correction, bool take_sqrt)
      : correction(correction), take_sqrt(take_sqrt) {}
};


// TODO: move to common.cuh, also upate layer_norm.cuh
template <typename T>
__device__ inline T Rsqrt(const T& x);

template <>
__device__ inline float Rsqrt(const float& x) {
  return rsqrtf(x);
}

template <>
__device__ inline double Rsqrt(const double& x) {
  return rsqrt(x);
}

template <>
__device__ inline half Rsqrt(const half& x) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return hrsqrt(x);
#else
  return half(rsqrtf(float(x)));
#endif
}




template <typename T>
__global__ void RowwiseMomentsCUDAKernel(
    int64_t N,
    double eps, // T eps,
    const T* X,
    T* mean,
    T* rstd) {
  using T_ACC = AccumulationType_t<T>; //acc_type<T, true>;
  using WelfordType = WelfordData<T_ACC, int64_t, T_ACC>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int64_t, T_ACC, thrust::pair<T_ACC, T_ACC>>;

  __shared__
      typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[C10_WARP_SIZE];
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);

  const int64_t i = blockIdx.x;
  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    val = welford_op.reduce(val, static_cast<T_ACC>(X[index]), index);
  }
  val = BlockReduce(
      val,
      welford_op,
      /*identity_element=*/WelfordType(0, 0, 0, 0),
      val_shared_ptr);

  if (threadIdx.x == 0) {
    T_ACC m1;
    T_ACC m2;
    thrust::tie(m2, m1) = welford_op.project(val);
    mean[i] = m1;
    rstd[i] = Rsqrt(m2 + static_cast<T_ACC>(eps));
  }
}

template <typename T>
__global__ void LayerNormForwardCUDAKernel(
    int64_t N,
    const T* X,
    const T* mean,
    const T* rstd,
    const T* gamma,
    const T* beta,
    T* Y) {
  using T_ACC =  AccumulationType_t<T>; //acc_type<T, true>;
  const int64_t i = blockIdx.x;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    const T_ACC gamma_v =
        gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[j]);
    const T_ACC beta_v =
        beta == nullptr ? T_ACC(0) : static_cast<T_ACC>(beta[j]);
    Y[index] = (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) *
            static_cast<T_ACC>(rstd[i]) * gamma_v + beta_v;
  }
}


template <typename T>
void LaunchLayerNorm(
    const cudaDeviceProp& prop,
    cudaStream_t stream,
    T* output,
    T* mean,
    T* inv_std_dev,
    const T* input,
    int n1,
    int n2,
    double epsilon,
    const T* gamma,
    const T* beta) {
  RowwiseMomentsCUDAKernel<T><<<n1, kCUDABlockReduceNumThreads, 0, stream>>>(
          n2, epsilon, input, mean, inv_std_dev);

  LayerNormForwardCUDAKernel<T><<<n1, kCUDANumThreads, 0, stream>>>(
      n2, input, mean, inv_std_dev, gamma, beta, output);
}

#define LAYERNORM_LINEAR_IMPL(T) \
  template void LaunchLayerNorm<T>(const cudaDeviceProp& prop, cudaStream_t stream, T* output, T* mean, T* inv_std_dev, const T* input, \
                                   int n1, int n2, double epsilon, const T* gamma, const T* beta);

LAYERNORM_LINEAR_IMPL(float)

/*
template <typename T>
void LayerNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    T eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  DCHECK_EQ(X.numel(), M * N);
  DCHECK(!gamma.defined() || gamma.numel() == N);
  DCHECK(!beta.defined() || beta.numel() == N);
  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.data_ptr<T>() : nullptr;
  T* Y_data = Y->data_ptr<T>();
  T* mean_data = mean->data_ptr<T>();
  T* rstd_data = rstd->data_ptr<T>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  RowwiseMomentsCUDAKernel<T>
      <<<M, kCUDABlockReduceNumThreads, 0, cuda_stream>>>(
          N, eps, X_data, mean_data, rstd_data);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  LayerNormForwardCUDAKernel<T><<<M, kCUDANumThreads, 0, cuda_stream>>>(
      N, X_data, mean_data, rstd_data, gamma_data, beta_data, Y_data);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}



void LayerNormKernelImpl(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    double eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "LayerNormKernelImpl",
      [&]() {
        LayerNormKernelImplInternal<scalar_t>(
            X, gamma, beta, M, N, static_cast<scalar_t>(eps), Y, mean, rstd);
      });
}

} // namespace
*/

/*
std::tuple<Tensor, Tensor, Tensor> layer_norm_cuda(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const c10::optional<Tensor>& weight_opt // optional,
    const c10::optional<Tensor>& bias_opt  // optional,
    double eps) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  Tensor Y = at::native::empty_like(
      *X,
      c10::nullopt, // dtype,
      c10::nullopt, // layout
      c10::nullopt, // device
      c10::nullopt, // pin_memory
      LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor mean = at::empty({M}, X->options());
  Tensor rstd = at::empty({M}, X->options());
  if (M > 0) {
    LayerNormKernelImpl(*X, *gamma, *beta, M, N, eps, &Y, &mean, &rstd);

    const auto input_shape = input.sizes();
    const size_t axis = input.dim() - normalized_shape.size();

    std::vector<int64_t> stat_shape;
    for (size_t idx = 0; idx < axis; ++idx) {
      stat_shape.push_back(input_shape[idx]);
    }
    for (size_t idx = axis; idx < input.dim(); ++idx) {
      stat_shape.push_back(1);
    }

    mean = mean.view(stat_shape);
    rstd = rstd.view(stat_shape);
  }
  return std::make_tuple(std::move(Y), std::move(mean), std::move(rstd));
}

REGISTER_DISPATCH(LayerNormKernel, &LayerNormKernelImpl);
*/

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
