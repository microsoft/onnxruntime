/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/
/*
Kernel implementation for Gamma rotary embeddings. 
This implementation below subgraph
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
#include <cmath>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/bert/gemma_rotary_emb_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;

template <typename T, typename U>
__global__ void GemmaRotaryEmb(
                                T* output1,
                                T* output2,
                                const U* emb,
                                const T* q,
                                const T* q_rot,
                                const T* k,
                                const T* k_rot,
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
      output1[qk_idx] = q[qk_idx] * cos_val + q_rot[qk_idx] * sin_val;
      output2[qk_idx] = k[qk_idx] * cos_val + k_rot[qk_idx] * sin_val;
    }
}

template <typename T, typename U>
Status LaunchGemmaRotaryEmbeddingKernel(
    cudaStream_t stream,
    T* output1,
    T* output2,
    const U* emb,
    const T* q,
    const T* q_rot,
    const T* k,
    const T* k_rot,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int dim
    ) {
  int blocksPerGrid = static_cast<int>(ceil(float(batch_size * num_heads * seq_len * dim) / kThreadsPerBlock));

  GemmaRotaryEmb<<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(
    output1, output2,
    emb, q, q_rot, k, k_rot,
    batch_size, num_heads, seq_len, dim
  );

  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchGemmaRotaryEmbeddingKernel<half, float>(
    cudaStream_t stream,
    half* output1,
    half* output2,
    const float* emb,
    const half* q,
    const half* q_rot,
    const half* k,
    const half* k_rot,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int dim);

template Status LaunchGemmaRotaryEmbeddingKernel<float, float>(
    cudaStream_t stream,
    float* output1,
    float* output2,
    const float* emb,
    const float* q,
    const float* q_rot,
    const float* k,
    const float* k_rot,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int dim);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
