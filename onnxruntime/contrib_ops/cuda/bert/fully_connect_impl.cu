// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "fully_connect_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, unsigned TPB>
__global__ void ExpandKernel(const int input_length, const T* input, T* output) {
  const int offset = blockIdx.x * input_length;
  for (int i = threadIdx.x; i < input_length; i += TPB) {
    const int idx = offset + i;
    const T val = input[i];
    output[idx] = val;
  }
}

template <typename T>
bool Expand(cudaStream_t stream, const int input_length, const int output_length, const T* input, T* output) {
  assert(output_length % input_length == 0);
  const int grid_size = output_length / input_length;
  constexpr int block_size = 256;
  ExpandKernel<T, block_size><<<grid_size, block_size, 0, stream>>>(input_length, input, output);
  return CUDA_CALL(cudaPeekAtLastError());
}

/*
bool LaunchExpandKernel(
    void* output,
    const void* input,
    const int input_length,
    const int output_length,
    const size_t element_size,
    cudaStream_t stream) {
  if (element_size == 2) {
    return Expand(stream, input_length, output_length,
                  reinterpret_cast<const half*>(input), reinterpret_cast<half*>(output));
  } else {
    return Expand(
        stream, input_length, output_length,
        reinterpret_cast<const float*>(input), reinterpret_cast<float*>(output));
  }
}
*/

bool launchFullyConnect(
    const void* input,
    const void* weights,
    const void* bias,
    void* output,
    const int batch_size,
    const int sequence_length,
    const int hidden_size,
    const int ld,
    cublasHandle_t& cublas,
    const size_t element_size,
    cudaStream_t stream) {
  int m = batch_size * sequence_length;
  int n = ld;
  //int n = 3 * hidden_size;
  int k = hidden_size;

//  if (!LaunchExpandKernel(output, bias, n, n * m, element_size, stream)) {
//    return false;
//  }

  if (element_size == 2) {
    if (!Expand(stream, n, n * m,
                reinterpret_cast<const half*>(bias), reinterpret_cast<half*>(output)))
      return false;

    half one = half(1.0f);
    //return CUBLAS_CALL(cublasGemmHelper(
    //    cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
    //    reinterpret_cast<const half*>(weights), n, reinterpret_cast<const half*>(input), k, 
    //    &one, reinterpret_cast<half*>(output), n));
    return CUBLAS_CALL(cublasHgemm(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
        reinterpret_cast<const half*>(weights), n, reinterpret_cast<const half*>(input), k,
        &one, reinterpret_cast<half*>(output), n));
  } else {
    if (!Expand(stream, n, n * m,
                reinterpret_cast<const float*>(bias), reinterpret_cast<float*>(output)))
      return false;
    float one = 1.0f;
    return CUBLAS_CALL(cublasSgemm(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
        reinterpret_cast<const float*>(weights), n, reinterpret_cast<const float*>(input), k,
        &one, reinterpret_cast<float*>(output), n));
    //return CUBLAS_CALL(cublasGemmHelper(
    //    cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
    //    reinterpret_cast<const float*>(weights), n, reinterpret_cast<const float*>(input), k, 
    //    &one, reinterpret_cast<float*>(output), n));
  }
  /*
  //P: BxNxSxS, V: BxNxSxH
  // compute P*V (as V*P), and store in scratch3: BxNxSxH
  if (!CUBLAS_CALL(CublasGemmStridedBatched(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, head_size, sequence_length, sequence_length, 1.f, v, head_size, size_per_batch,
          scratch2, sequence_length, temp_matrix_size, 0.f, scratch3, head_size, size_per_batch, batches))) {
    return false;
  }
  */
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
