/*
 The implementation of this file is based on skipLayerNorm plugin in TensorRT demo:
 https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT/

Copyright 2019 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/layer_norm.cuh"
#include "contrib_ops/cuda/bert/skip_layer_norm_impl.h"
#include <cuda_fp16.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr float one = 1.0;

template <typename T, unsigned TPB>
__global__ void SkipLayerNormKernelSmall(
    const int ld, const T* input, const T* skip, const T* beta, const T* gamma, const T* bias,
    const T epsilon, T* output) {
  const T reverse_ld = T(1.f / ld);
  const int offset = blockIdx.x * ld;

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  cub::KeyValuePair<T, T> thread_data(0, 0);
  const int idx = offset + threadIdx.x;
  T val = 0;

  if (threadIdx.x < ld) {
    val = (bias == nullptr) ? input[idx] + skip[idx] : input[idx] + skip[idx] + bias[threadIdx.x];
    const T rldval = reverse_ld * val;
    thread_data = pair_sum(thread_data, cub::KeyValuePair<T, T>(rldval, rldval * val));
  }

  LayerNormSmall<T, TPB>(val, thread_data, ld, idx, beta, gamma, epsilon, output);
}

template <typename T, unsigned TPB> // TODO: T is redundant here!
__global__ void SkipLayerNormKernelSmall2(
    const int ld, const half2* input, const half2* skip, const half2* beta,
    const half2* gamma, const half2* bias, const half2 epsilon, half2* output) {
  // const half2 reverse_ld = T(1.f / ld);
  //const half2 reverse_ld = h2rcp(__float2half2_rn(float(ld))); // TODO

  // workaround for a llvm bug: https://github.com/intel/llvm/issues/5153
  const half2 one2 = __float2half2_rn(one);
  const half2 ld2 = __float2half2_rn(float(ld));
  const half2 reverse_ld = one2 / ld2;

  /*
  const half2 ld2 = __float2half2_rn(float(ld));
  const half2 reverse_ld = h2rcp(ld2);
  */
  const int offset = blockIdx.x * ld; // shall I refactor this offset

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  // cub::KeyValuePair<half2, half2> thread_data(0, 0); // TODO: How to initialize a half2 pair
  cub::KeyValuePair<half2, half2> thread_data(__float2half2_rn(float(0.0)), __float2half2_rn(float(0.0))); // TODO: How to initialize a half2 pair
  const int idx = offset + threadIdx.x;
  half2 val = __float2half2_rn(float(0.0));

  if (threadIdx.x < ld) {
    val = (bias == nullptr) ? input[idx] + skip[idx] : input[idx] + skip[idx] + bias[threadIdx.x];
    const half2 rldval = reverse_ld * val;
    thread_data = pair_sum(thread_data, cub::KeyValuePair<half2, half2>(rldval, rldval * val));
  }

  LayerNormSmall<half2, TPB>(val, thread_data, ld, idx, beta, gamma, epsilon, output);
}

template <typename T, unsigned TPB>
__global__ void SkipLayerNormKernel(
    const int ld, const T* input, const T* skip, const T* beta, const T* gamma, const T* bias,
    const T epsilon, T* output) {
  const T reverse_ld = T(1.f / ld);
  const int offset = blockIdx.x * ld;

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  cub::KeyValuePair<T, T> thread_data(0, 0);

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const T val = (bias == nullptr) ? input[idx] + skip[idx] : input[idx] + skip[idx] + bias[i];
    const T rldval = reverse_ld * val;
    thread_data = pair_sum(thread_data, cub::KeyValuePair<T, T>(rldval, rldval * val));
    output[idx] = val;
  }

  LayerNorm<T, TPB>(thread_data, ld, offset, beta, gamma, epsilon, output);
}

template <typename T, unsigned TPB> // TODO: T is redundant here!
__global__ void SkipLayerNormKernel2(
    const int ld, const half2* input, const half2* skip, const half2* beta,
    const half2* gamma, const half2* bias, const half2 epsilon, half2* output) {
  // const half2 reverse_ld = T(1.f / ld);
  //const half2 reverse_ld = h2rcp(__float2half2_rn(float(ld))); // TODO
  const half2 one2 = __float2half2_rn(one);
  const half2 ld2 = __float2half2_rn(float(ld));
  const half2 reverse_ld = one2 / ld2;
  const int offset = blockIdx.x * ld; // shall I refactor this offset

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  // cub::KeyValuePair<half2, half2> thread_data(0, 0); // TODO: How to initialize a half2 pair
  cub::KeyValuePair<half2, half2> thread_data(__float2half2_rn(float(0.0)), __float2half2_rn(float(0.0))); // TODO: How to initialize a half2 pair
  const int idx = offset + threadIdx.x;
  half2 val = __float2half2_rn(float(0.0)); // TODO: Can I initialize half2 like this?

  for (int i = threadIdx.x; i < ld; i += TPB) {
    // val = (bias == nullptr) ? input[idx] + skip[idx] : input[idx] + skip[idx] + bias[threadIdx.x];
    // const half2 rldval = reverse_ld * val;
    // thread_data = pair_sum(thread_data, cub::KeyValuePair<half2, half2>(rldval, rldval * val));
    const int idx = offset + i;
    const half2 val = (bias == nullptr) ? input[idx] + skip[idx] : input[idx] + skip[idx] + bias[i];
    const half2 rldval = reverse_ld * val;
    thread_data = pair_sum(thread_data, cub::KeyValuePair<half2, half2>(rldval, rldval * val));
    output[idx] = val;
  }

  LayerNormSmall<half2, TPB>(val, thread_data, ld, idx, beta, gamma, epsilon, output);
}

// template <typename T, unsigned TPB, int VPT>
template <unsigned TPB>
__global__ void skipln_vec_2(
  const int ld, const half2* input, const half2* skip, const half2* beta, const half2* gamma, const half2* bias, half epsilon, half2* output, bool hasBias) {
  const int idx = TPB * blockIdx.x + threadIdx.x; // TPB --> ld
  // 4 * 1024 * 4 * 2 Bytes = 16KB per block

//   T in_local[VPT];
//   T skip_local[VPT];
//   T bias_local[VPT];
//   // T gamma_local[VPT];
//   copy<sizeof(T) * VPT>(&input[idx], in_local); // input[ld * blockIdx.x + threadIdx.x * VPT]
//   copy<sizeof(T) * VPT>(&skip[idx], skip_local);// skip[ld * blockIdx.x + threadIdx.x * VPT]
//   copy<sizeof(T) * VPT>(&bias[threadIdx.x * VPT], bias_local); // bias[threadIdx.x * VPT]
  half2 input_vec = input[idx];
  const half2 skip_vec = skip[idx];
  const half2 beta_vec = (beta == nullptr) ? __float2half2_rn(0.f) : beta[threadIdx.x];
  const half2 gamma_vec = gamma[threadIdx.x];
  const half2 bias_vec = (hasBias) ? bias[threadIdx.x] : __float2half2_rn(0.f);
  const half2 rld = __float2half2_rn(1.f / ld);

  input_vec += skip_vec;
  if (hasBias) {
    input_vec += bias_vec;
  }
  const half2 tmp = rld * input_vec;
  // const half2 tmp = input_vec;
  //const half2 local = rld * input_vec;
  //const half2 local2 = tmp * input_vec;

//   #pragma unroll
//   for (int it = 0; it < 2; it++) {
//     in_local[it] += skip_local[it];
//     if (hasBias) {
//       in_local[it] += bias_local[it];
//     }
//     const T tmp = rld * in_local[it];
//     local += tmp;
//     local2 += tmp * in_local[it];
//   }

//   copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], bias_local);
//   copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], skip_local);
  cub::KeyValuePair<half2, half2> thread_data(tmp,  tmp * input_vec); // x, (x^2)
  // cub::KeyValuePair<half2, half2> thread_data(input_vec,  input_vec * input_vec); // x, (x^2)
  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<half2, half2>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  //__shared__ half2 mu;     // mean
  //__shared__ half2 rsigma; // 1 / std.dev.
  __shared__ half mu;     // mean
  __shared__ half rsigma; // 1 / std.dev.

  /*  half local = rld * (input_vec.x + input_vec.y); */
  /*
  half local = rld * (input_vec.x + input_vec.y);
  half local2 = rld * input_vec.x * input_vec.x + rld * input_vec.y * input_vec.y;
  cub::KeyValuePair<half, half> thread_data(local,local2);
  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<half, half>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ half mu;     // mean
  __shared__ half rsigma; // 1 / std.dev.
  */
  KeyValuePairSum pair_sum;
  //const auto sumKV = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);
  //const cub::KeyValuePair<half, half> sumKV = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);
  const cub::KeyValuePair<half2, half2> sumKV = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);

  if (threadIdx.x == 0) {
    mu = (sumKV.key.x + sumKV.key.y); // half  sum(x/n)
    half temp = sumKV.value.x + sumKV.value.y; // sum(x^2/n)
    rsigma = Rsqrt(temp - mu * mu + epsilon); // half --> sumKV.value.x + sumKV.value.y - mu * mu + epsilon
    //rsigma = Rsqrt(sumKV.value.x + sumKV.value.y - mu * mu + epsilon); // this does not work!
    //mu = sumKV.key;
    //rsigma = Rsqrt(sumKV.value - mu * mu + epsilon);
    //rsigma = Rsqrt(sumKV.value - mu * mu);
  } //(1, 2, 4) --> 
  __syncthreads();


  half2 val = input_vec - __half2half2(mu);
  ///workspace/pr_skiplayernorm/onnxruntime/build/RelWithDebInfo/amdgpu/onnxruntime/contrib_ops/rocm/bert/skip_layer_norm_impl.cu:221:31: error: use of overloaded operator '*' is ambiguous (with operand types 'int' and 'half' (aka '__half'))
  //half output_vec = gamma_vec.x * val.x * rsigma + beta_vec.x;
  half2 output_vec = gamma_vec * val * __half2half2(rsigma) + beta_vec;
  //val.x = gamma_vec.x * val.x * rsigma + beta_vec.x
  //val.x = gamma_vec.x * val.x * rsigma + beta_vec.x;
  //val.y = gamma_vec.y * val.y * rsigma + beta_vec.y;
  output[idx] = output_vec;
  //
  //half2 output_vec = gamma_vec * (input_vec - mu) * rsigma + beta_vec;
  //half2 output_vec = gamma_vec * (input_vec - __half2half2(mu)) * __half2half2(rsigma) + beta_vec;

//   copy<sizeof(T) * VPT>(in_local, &output[idx]);

  //half2 output_vec = __float2half2_rn(0.f);
  //output[idx] = output_vec;
  //output[idx].x = gamma_vec.x * (input_vec.x - mu) * rsigma + beta_vec.x;
  //output[idx].x = gamma_vec.y * (input_vec.y - mu) * rsigma + beta_vec.y;
}

template <unsigned TPB>
__global__ void skipln_vec_2_new(
  const int ld, const half2* input, const half2* skip, const half2* beta, const half2* gamma, const half2* bias, half epsilon, half2* output, bool hasBias) {
  const int idx = TPB * blockIdx.x + threadIdx.x; // TPB --> ld
  // 4 * 1024 * 4 * 2 Bytes = 16KB per block

//   T in_local[VPT];
//   T skip_local[VPT];
//   T bias_local[VPT];
//   // T gamma_local[VPT];
//   copy<sizeof(T) * VPT>(&input[idx], in_local); // input[ld * blockIdx.x + threadIdx.x * VPT]
//   copy<sizeof(T) * VPT>(&skip[idx], skip_local);// skip[ld * blockIdx.x + threadIdx.x * VPT]
//   copy<sizeof(T) * VPT>(&bias[threadIdx.x * VPT], bias_local); // bias[threadIdx.x * VPT]
  half2 input_vec = input[idx];
  const half2 skip_vec = skip[idx];
  half2 beta_vec = (beta == nullptr) ? __float2half2_rn(0.f) : beta[threadIdx.x];
  const half2 gamma_vec = gamma[threadIdx.x];
  half2 bias_vec = (hasBias) ? bias[threadIdx.x] : __float2half2_rn(0.f);
  // const half2 rld = __float2half2_rn(1.f / ld);
  const half rld = half(1.f / ld);

  input_vec += skip_vec;
  if (hasBias) {
    input_vec += bias_vec;
  }
  //const half tmp = rld * (input_vec.x + input_vec.y);
  
  cub::KeyValuePair<half, half> thread_data(__hmul(rld, __hadd(input_vec.x, input_vec.y)),
                                            __hadd(__hmul(rld, __hmul(input_vec.x, input_vec.x)),
                                            __hmul(rld, __hmul(input_vec.y, input_vec.y))));
                                            //rld * (input_vec.x + input_vec.x) + rld * (input_vec.y + input_vec.y)); // x, (x^2)
  // cub::KeyValuePair<half2, half2> thread_data(input_vec,  input_vec * input_vec); // x, (x^2)
  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<half, half>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  //__shared__ half2 mu;     // mean
  //__shared__ half2 rsigma; // 1 / std.dev.
  __shared__ half mu;     // mean
  __shared__ half rsigma; // 1 / std.dev.

  KeyValuePairSum pair_sum;
  //const auto sumKV = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);
  //const cub::KeyValuePair<half, half> sumKV = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);
  const cub::KeyValuePair<half, half> sumKV = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);

  if (threadIdx.x == 0) {
    mu = sumKV.key;
    // half temp = sumKV.value.x + sumKV.value.y; // sum(x^2/n)
    rsigma = Rsqrt(sumKV.value - mu * mu + epsilon); // half --> sumKV.value.x + sumKV.value.y - mu * mu + epsilon
    //rsigma = Rsqrt(sumKV.value.x + sumKV.value.y - mu * mu + epsilon); // this does not work!
    //mu = sumKV.key;
    //rsigma = Rsqrt(sumKV.value - mu * mu + epsilon);
    //rsigma = Rsqrt(sumKV.value - mu * mu);
  } //(1, 2, 4) --> 
  __syncthreads();

  half2 val = input_vec - __half2half2(mu);
  half2 output_vec;
  /* 
  /workspace/pr_skiplayernorm/onnxruntime/build/RelWithDebInfo/amdgpu/onnxruntime/contrib_ops/rocm/bert/skip_layer_norm_impl.cu:301:38: error: use of overloaded operator '*' is ambiguous (with operand types 'int' and 'half' (aka '__half'))
  output_vec.y = gamma_vec.y * val.y * rsigma + beta_vec.y;
                 ~~~~~~~~~~~~~~~~~~~ ^ ~~~~~~
/opt/rocm-5.0.1/hip/include/hip/amd_detail/amd_hip_fp16.h:278:24: note: candidate function
                __half operator*(const __half& x, const __half& y)
                       ^
/workspace/pr_skiplayernorm/onnxruntime/build/RelWithDebInfo/amdgpu/onnxruntime/contrib_ops/rocm/bert/skip_layer_norm_impl.cu:301:38: note: built-in candidate operator*(int, float)
  output_vec.y = gamma_vec.y * val.y * rsigma + beta_vec.y;
                                     ^



  output_vec.x = gamma_vec.x * val.x * rsigma + beta_vec.x;
  output_vec.y = gamma_vec.y * val.y * rsigma + beta_vec.y;
  */
  /*
   /workspace/pr_skiplayernorm/onnxruntime/build/RelWithDebInfo/amdgpu/onnxruntime/contrib_ops/rocm/bert/skip_layer_norm_impl.cu:301:32: 
       error: call to '__hadd' is ambiguous
       output_vec.y = gamma_vec.y * __hadd(input_vec.y, -half(mu)) * rsigma + beta_vec.y;
                               ^~~~~~
       /opt/rocm-5.0.1/hip/include/hip/amd_detail/amd_device_functions.h:184:39: note: candidate function
       __device__ static inline unsigned int __hadd(int x, int y) {
                                      ^
       /opt/rocm-5.0.1/hip/include/hip/amd_detail/amd_hip_fp16.h:1268:20: note: candidate function
            __half __hadd(__half x, __half y)
                   ^
   1 warning and 2 errors generated when compiling for gfx906.
   
   */

  //output_vec.y = gamma_vec.y * __hadd(input_vec.y, -half(mu)) * rsigma + beta_vec.y;
  ///workspace/pr_skiplayernorm/onnxruntime/build/RelWithDebInfo/amdgpu/onnxruntime/contrib_ops/rocm/bert/skip_layer_norm_impl.cu:221:31: error: use of overloaded operator '*' is ambiguous (with operand types 'int' and 'half' (aka '__half'))
  //half output_vec = gamma_vec.x * val.x * rsigma + beta_vec.x;
  
  //half2 output_vec = gamma_vec * val * __half2half2(rsigma) + beta_vec;
  //val.x = gamma_vec.x * val.x * rsigma + beta_vec.x
  //val.x = gamma_vec.x * val.x * rsigma + beta_vec.x;
  //val.y = gamma_vec.y * val.y * rsigma + beta_vec.y;
  output[idx] = output_vec;
}

template <unsigned TPB>
__global__ void skipln_vec(
  const int ld, const half* input, const half* skip, const half* beta, const half* gamma, const half* bias, half epsilon, half* output, bool hasBias) {
  const int idx = TPB * blockIdx.x + threadIdx.x; // TPB --> ld
  // 4 * 1024 * 4 * 2 Bytes = 16KB per block
  half input_vec = input[idx];
  const half skip_vec = skip[idx];
  const half beta_vec = (beta == nullptr) ? half(0.f) : beta[threadIdx.x];
  const half gamma_vec = gamma[threadIdx.x];
  const half bias_vec = (hasBias) ? bias[threadIdx.x] : half(0.f);
  const half rld = half(1.f / ld);

  input_vec += skip_vec;
  if (hasBias) {
    input_vec += bias_vec;
  }
  const half tmp = rld * input_vec;

  cub::KeyValuePair<half, half> thread_data(tmp,  tmp * input_vec);
  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<half, half>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ half mu;     // mean
  __shared__ half rsigma; // 1 / std.dev.
  KeyValuePairSum pair_sum;
  //const auto sumKV = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);
  //const cub::KeyValuePair<half, half> sumKV = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);
  const cub::KeyValuePair<half, half> sumKV = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);

  if (threadIdx.x == 0) {
    mu = sumKV.key; // half
    half temp = sumKV.value;
    rsigma = Rsqrt(temp - mu * mu + epsilon);
  }
  __syncthreads();
  half val = input_vec - mu;
  half output_vec = gamma_vec * val * rsigma + beta_vec;
  output[idx] = output_vec;
}

/* float32 */
bool ComputeSkipLayerNorm(
    const cudaDeviceProp& prop, cudaStream_t stream, const int ld, const int n, const float* input,
    const float* skip, const float* beta, const float* gamma, const float* bias, const float epsilon, float* output, bool use_half2) {
  // this must be true because n is the total size of the tensor
  assert(n % ld == 0);
  const int grid_size = n / ld;

  if (ld <= 32) {
    constexpr int block_size = 32;
    SkipLayerNormKernelSmall<float, block_size>
        <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
  } else if (ld <= 128) {
    constexpr int block_size = 128;
    SkipLayerNormKernelSmall<float, block_size>
        <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
  } else if (ld == 384) {
    constexpr int block_size = 384;
    SkipLayerNormKernelSmall<float, block_size>
        <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
  } else {
    constexpr int block_size = 256;
    SkipLayerNormKernel<float, block_size><<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
  }
  return CUDA_CALL(cudaPeekAtLastError());
}

/* half16 */
bool ComputeSkipLayerNorm(
    const cudaDeviceProp& prop, cudaStream_t stream, const int ld, const int n, const half* input,
    const half* skip, const half* beta, const half* gamma, const half* bias, const half epsilon, half* output, bool use_half2) {
  // this must be true because n is the total size of the tensor
  assert(n % ld == 0);
  if (use_half2 && 0 == (n & 1) && prop.major >= 7) {
    const int grid_size = n / ld;

    const half2* input2 = reinterpret_cast<const half2*>(input);
    const half2* skip2 = reinterpret_cast<const half2*>(skip);
    const half2* beta2 = reinterpret_cast<const half2*>(beta);
    const half2* gamma2 = reinterpret_cast<const half2*>(gamma);
    const half2* bias2 = reinterpret_cast<const half2*>(bias);
    half2* output2 = reinterpret_cast<half2*>(output);
    const half2 epsilon2 = __half2half2(epsilon);
    constexpr int VPT = 32 / sizeof(half); // 16 (og)
    bool hasBias = (bias == nullptr) ? false : true; // TODO: template args (define in .cc file)

    if (ld <= 32) {
    //   constexpr int block_size = 32;
    //   SkipLayerNormKernelSmall2<half, block_size>
    //       <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);

      // constexpr int block_size = 4 / 2; // for testing (hipcub --> VPT needs to be defined explicitly)
      // skipln_vec_2<block_size>
      //     <<<grid_size, block_size, 0, stream>>>(ld, input2, skip2, beta2, gamma2, bias2, epsilon, output2, hasBias);
      // constexpr int block_size = 4; // for testing (hipcub --> VPT needs to be defined explicitly)
      // skipln_vec<block_size>
      //     <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output, hasBias);
      constexpr int block_size = 4 / 2; // for testing (hipcub --> VPT needs to be defined explicitly)
      skipln_vec_2_new<block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input2, skip2, beta2, gamma2, bias2, epsilon, output2, hasBias);
    } else if (ld == 128) {
      constexpr int block_size = 128 / 2; // for testing
      skipln_vec_2<block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input2, skip2, beta2, gamma2, bias2, epsilon, output2, hasBias);
    } else if (ld == 384) {
      constexpr int block_size = 384 / 2; // for testing
      skipln_vec_2<block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input2, skip2, beta2, gamma2, bias2, epsilon, output2, hasBias);
    } else if (ld == 1024) {
      constexpr int block_size = 1024 / 2; // for testing
      skipln_vec_2<block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input2, skip2, beta2, gamma2, bias2, epsilon, output2, hasBias);
    } else {
      // TODO: check if half2 also works for this function or not
      constexpr int block_size = 256;
      SkipLayerNormKernel<half, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
    }
  } else {
    const int grid_size = n / ld;
    if (ld <= 32) {
      constexpr int block_size = 32;
      SkipLayerNormKernelSmall<half, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
    } else if (ld <= 128) {
      constexpr int block_size = 128;
      SkipLayerNormKernelSmall<half, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
    } else if (ld == 384) {
      constexpr int block_size = 384;
      SkipLayerNormKernelSmall<half, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
    } else {
      constexpr int block_size = 256;
      SkipLayerNormKernel<half, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
    }
  }
  return CUDA_CALL(cudaPeekAtLastError());
}

template<>
bool LaunchSkipLayerNormKernel(
    const cudaDeviceProp& prop,
    cudaStream_t stream,
    half* output,
    const half* input,
    const half* skip,
    const half* gamma,
    const half* beta,
    const half* bias,
    float epsilon,
    int hidden_size,
    int element_count,
    size_t element_size,
    bool use_half2) {

  return ComputeSkipLayerNorm(
         prop,
         stream,
         hidden_size,
         element_count,
         reinterpret_cast<const half*>(input),
         reinterpret_cast<const half*>(skip),
         reinterpret_cast<const half*>(beta),
         reinterpret_cast<const half*>(gamma),
         reinterpret_cast<const half*>(bias),
         __float2half_rn(epsilon),
         reinterpret_cast<half*>(output),
         use_half2);
}

template<>
bool LaunchSkipLayerNormKernel(
    const cudaDeviceProp& prop,
    cudaStream_t stream,
    float* output,
    const float* input,
    const float* skip,
    const float* gamma,
    const float* beta,
    const float* bias,
    float epsilon,
    int hidden_size,
    int element_count,
    size_t element_size,
    bool use_half2) {

  return ComputeSkipLayerNorm(
         prop,
         stream,
         hidden_size,
         element_count,
         reinterpret_cast<const float*>(input),
         reinterpret_cast<const float*>(skip),
         reinterpret_cast<const float*>(beta),
         reinterpret_cast<const float*>(gamma),
         reinterpret_cast<const float*>(bias),
         epsilon,
         reinterpret_cast<float*>(output),
         false);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

