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
#include "contrib_ops/cuda/bert/gemma_rotary_emb_impl.h"
#include <cmath>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, typename U>
__global__ void GemmaRotaryEmb( T* output1,
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
    const int t = blockIdx.x;
    const int x = blockIdx.y;
    const int y = blockIdx.z;
    const int z = threadIdx.x;

    if (t < batch_size && x < num_heads && y < seq_len && z < dim) {
        // Calculate linear indices for accessing elements in the flattened tensors
        int emb_idx = t * num_heads * dim + y * dim + z;
        int qk_idx = t * num_heads * seq_len * dim + x * seq_len * dim + y * dim + z;
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

  const dim3 block(dim);
  const dim3 grid(batch_size, num_heads, seq_len);

  GemmaRotaryEmb<<<grid, block, 0, stream>>>(
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

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
