/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/

/*
Kernel implementation for Gamma rotary embeddings grad.
This implementation below subgraph
Grad kernel inputs/outputs (grad kernel mapping inputs/outputs in gradient_builder.cc)
I0 (GO0): dY1
I1 (GO1): dY2
I2 (I0): X
O0 (GI1): dq
O1 (GI2): dq_rot
O2 (GI3): dk
O3 (GI4): dk_rot
                   I2
                /      \
              Sin      Cos
               |        |
              Cast     Cast
              |---------|------------|
  |-----------|---------|            |
  |       I0  |         |      I1    |
  |     /    \|         |     /   \  |
Mul_Grad   Mul_Grad   Mul_Grad   Mul_Grad
  |           |          |           |
  O0         O1          O2          O3
*/

#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "orttraining/training_ops/cuda/math/gemma_rotary_emb_grad_impl.h"
#include <cmath>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace cuda {

// constexpr int kElementsPerThread = GridDim::maxElementsPerThread;
constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;

template <typename T, typename U>
__global__ void GemmaRotaryEmbeddingGrad(
                                    T* q_grad, 
                                    T* q_rot_grad, 
                                    T* k_grad, 
                                    T* k_rot_grad, 
                                    const T* go0, 
                                    const T* go1, 
                                    const U* emb, 
                                    const int batch_size, 
                                    const int num_heads, 
                                    const int seq_len, 
                                    const int dim) {
    const int qk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // index [i, j, k, l] -> [i, k, l]
    const int emb_idx = qk_idx / (num_heads * seq_len * dim) * (seq_len * dim) + qk_idx %  (seq_len * dim);
    if (qk_idx < batch_size * num_heads * seq_len * dim) {
        T sin_val = static_cast<T>(sin(emb[emb_idx])); 
        T cos_val = static_cast<T>(cos(emb[emb_idx]));
        // MLFloat16 k_grad_output_sum = go1[qk_idx] + go2[qk_idx];
        q_grad[qk_idx] = go0[qk_idx] * cos_val;
        q_rot_grad[qk_idx] = go0[qk_idx] * sin_val;
        k_grad[qk_idx] = go1[qk_idx] * cos_val;
        k_rot_grad[qk_idx] = go1[qk_idx] * sin_val;
    }
}

template <typename T, typename U>
Status LaunchGemmaRotaryEmbeddingGradKernel(
    cudaStream_t stream,
    T* q_grad,
    T* q_rot_grad,
    T* k_grad,
    T* k_rot_grad,
    const T* go0,
    const T* go1,
    const U* emb,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int dim
    ) {
  int blocksPerGrid = static_cast<int>(ceil(float(batch_size * num_heads * seq_len * dim) / kThreadsPerBlock));

  GemmaRotaryEmbeddingGrad<<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(
    q_grad, q_rot_grad, k_grad, k_rot_grad,
    go0, go1, emb,
    batch_size, num_heads, seq_len, dim
  );

  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchGemmaRotaryEmbeddingGradKernel<half, float>(
    cudaStream_t stream,
    half* q_grad,
    half* q_rot_grad,
    half* k_grad,
    half* k_rot_grad,
    const half* go0,
    const half* go1,
    const float* emb,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int dim);

template Status LaunchGemmaRotaryEmbeddingGradKernel<float, float>(
    cudaStream_t stream,
    float* q_grad,
    float* q_rot_grad,
    float* k_grad,
    float* k_rot_grad,
    const float* go0,
    const float* go1,
    const float* emb,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int dim);

}  // namespace cuda
}  // namespace onnxruntime
