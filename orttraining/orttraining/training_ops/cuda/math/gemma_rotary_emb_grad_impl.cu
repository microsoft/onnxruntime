/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/

/*
Kernel implementation for Gamma rotary embeddings.
This implementation below subgraph TODO
           (emb)
          /   \
        /      \
     Sin         Cos
      |             |
     Cast           Cast
      |              |
  Unsqueeze        Unsqueeze
 \/        \/   \/         \/
 Mul       Mul   Mul        Mul
    \     /         \     /
      Add             Add  
       |               |
    (output1)         (output2)
*/

#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/bert/gemma_rotary_emb_grad_impl.h"
#include <cmath>

using namespace onnxruntime::cuda;
namespace onnxruntime {
namespace contrib {
namespace cuda {

// constexpr int kElementsPerThread = GridDim::maxElementsPerThread;
constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;

template <typename T, typename U>
__global__ void gemmaRotaryEmbGrad(
                                    MLFloat16 *go0, 
                                    MLFloat16 *go1, 
                                    MLFloat16 *go2, 
                                    float *emb, 
                                    MLFloat16 *q_grad, 
                                    MLFloat16 *q_rot_grad, 
                                    MLFloat16 *k_grad, 
                                    MLFloat16 *k_rot_grad, 
                                    int batch_size, 
                                    int num_heads, 
                                    int seq_len, 
                                    int dim) {
    const int qk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // index [i, j, k, l] -> [i, k, l]
    const int emb_idx = qk_idx / (num_heads * seq_len * dim) * (seq_len * dim) + qk_idx %  (seq_len * dim);
    if (qk_idx < batch_size * num_heads * seq_len * dim) {
        MLFloat16 sin_val = static_cast<T>(sin(emb[emb_idx])); 
        MLFloat16 cos_val = static_cast<T>(cos(emb[emb_idx]));
        MLFloat16 k_grad_output_sum = go1[qk_idx] + go2[qk_idx];
        q_grad = go0[qk_idx] * cos_val;
        q_rot_grad = go0[qk_idx] * sin_val;
        k_grad = k_grad_output_sum * cos_val;
        k_rot_grad = k_grad_output_sum * sin_val;
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
    const T* go2, 
    const U* emb,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int dim
    ) {
  int blocksPerGrid = static_cast<int>(ceil(float(batch_size * num_heads * seq_len * dim) / kThreadsPerBlock));

  GemmaRotaryEmbGrad<<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(
    q_grad, q_rot_grad, k_grad, k_rot_grad,
    go0, go1, go2, emb,
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
    const half* go2,
    const float* emb,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int dim);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime