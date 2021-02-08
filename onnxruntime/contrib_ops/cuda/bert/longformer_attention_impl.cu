/*
Copyright (c) NVIDIA Corporation and Microsoft Corporation

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

// Limitations of current Longformer Attention CUDA Kernels:
// (1) Does not support global tokens in the middle. All global tokens shall be in the beginning of sequence.
// (2) Batch size <= 128 (defined in MAX_LONGFORMER_BATCH_SIZE)

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <library_types.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "longformer_attention_impl.h"
#include "attention_impl.h"
#include "attention_softmax.h"

using namespace onnxruntime::cuda;
using namespace cub;

#define CHECK(status)         \
  if (!CUBLAS_CALL(status)) { \
    return false;             \
  }

constexpr int MAX_LONGFORMER_BATCH_SIZE = 128;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Denote: batch size (B), sequence length (S), number of heads (N), dimension per head (H), max number of global tokens (G)
//
// Workspace layout (by default, the data type T is float or half):
//   [SoftmaxSpace: see below] [Q:BxNxSxH] [K:BxNxSxH] [V:BxNxSxH] [Global_Q:BxNxGxH] [Global_K:BxNxSxH] [Global_V:BxNxSxH]
// where Global_Q, Global_K and Global_V are optional. They are not allocated when there is no any global token.
//
// SoftmaxSpace layout (tmp_storage could use the space of scratch1, scratch2, Q and K):
//   [Global_Idx: int BxS][batch_global_num: int BxS][sequence_index: int BxS][tmp_storage: int 1024x1]
//                                                                            [scratch1: BxNxSxS ] [scratch2: BxNxSxS ]
// Allocated size could be slightly larger than needed: batch_global_num uses only Bx1 and allocated BxS.
// Scratch size is allocated as multiples of 256.

size_t GetLongformerSoftmaxWorkspaceSize(
    size_t element_size,
    int batch_size,
    int num_heads,
    int sequence_length) {
  size_t temp_size = sizeof(int) * 1024;
  size_t scratch_size = 2 * GetAttentionScratchSize(element_size, batch_size, num_heads, sequence_length, sequence_length);
  return 3 * batch_size * sequence_length * sizeof(int) + std::max(scratch_size, temp_size);
}

size_t GetLongformerAttentionWorkspaceSize(
    size_t element_size,
    int batch_size,
    int num_heads,
    int head_size,
    int sequence_length,
    int max_num_global) {
  size_t softmax_size = GetLongformerSoftmaxWorkspaceSize(element_size, batch_size, num_heads, sequence_length);
  size_t qkv_size = 3 * batch_size * sequence_length * num_heads * head_size * element_size;
  size_t global_qkv_size = max_num_global > 0 ? qkv_size : 0;
  return softmax_size + qkv_size + global_qkv_size;
}

__global__ void InitSequenceIndexKernel(int* sequence_index, int sequence_length) {
  int batch_index = blockIdx.x;
  for (int i = threadIdx.x; i < sequence_length; i += blockDim.x) {
    sequence_index[batch_index * sequence_length + i] = i;
  }
}

// TODO: Move this to its own plugin that can be run once for all layers.
int* BuildGlobalIndex(cudaStream_t stream, const int* global_attention, int batch_size, int sequence_length, void* workspace, size_t softmax_workspace_size) {
  int* global_idx = reinterpret_cast<int*>(workspace);
  int* batch_global_num = global_idx + batch_size * sequence_length;  // Number of global tokens in each batch, shape is (batch_size)
  int* sequence_index = batch_global_num + batch_size * sequence_length;
  int* tmp_storage = sequence_index + batch_size * sequence_length;

  InitSequenceIndexKernel<<<batch_size, 128, 0, stream>>>(sequence_index, sequence_length);

  // Determine temporary device storage requirements
  // Find the global attention indices and number of global attention tokens
  size_t temp_storage_bytes = 0;
  cub::DevicePartition::Flagged(NULL, temp_storage_bytes, sequence_index,
                                global_attention, global_idx, batch_global_num, sequence_length, stream);
  assert(temp_storage_bytes <= softmax_workspace_size - static_cast<size_t>(3 * batch_size * sequence_length));

  for (int i = 0; i < batch_size; ++i) {
    cub::DevicePartition::Flagged(reinterpret_cast<void*>(tmp_storage), temp_storage_bytes, sequence_index,
                                  global_attention + i * sequence_length, global_idx + i * sequence_length,
                                  batch_global_num + i, sequence_length, stream);
  }

  return global_idx;
}

template <typename T, int blockSize>
__launch_bounds__(blockSize)
    __global__ void LongformerSoftmaxKernel(const int* global_attention,
                                            const int* global_idx,
                                            const int* batch_global_num,
                                            const T* input,
                                            const T* attention_mask,
                                            T* output,
                                            float scaler,
                                            int dim0,
                                            int sequence_length,
                                            int attention_window) {
  typedef cub::BlockReduce<float, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage block_reduce_temp;
  __shared__ float max_shared;
  __shared__ float sum_shared;

  const T* input_block = input + sequence_length * blockIdx.x;
  T* output_block = output + sequence_length * blockIdx.x;
  const int batch_index = blockIdx.x / dim0;
  const int row_index = blockIdx.x % sequence_length;
  const int global_num = batch_global_num[batch_index];

  // To be consistent with Huggingface Longformer, the row of maksed word are set as zero.
  if ((float)attention_mask[batch_index * sequence_length + row_index] < 0.0f) {
    for (int i = threadIdx.x; i < sequence_length; i += blockSize) {
      output_block[i] = (T)(0);
    }
    return;
  }

  // local attention token
  int col_start = 0;
  int col_end = sequence_length;
  bool is_local_row = (global_attention[batch_index * sequence_length + row_index] == (int)0);
  if (is_local_row) {
    col_start = row_index - attention_window;
    if (col_start < 0) {
      col_start = 0;
    }

    col_end = row_index + attention_window + 1;
    if (col_end > sequence_length) {
      col_end = sequence_length;
    }
  }

  const T* mask_block = attention_mask + sequence_length * batch_index;
  int tid = threadIdx.x;

  // calculate max input
  float max_input = -CUDART_INF_F;
  // #pragma unroll 16
  for (int i = tid + col_start; i < col_end; i += blockSize) {
    float x = input_block[i];
    x = x * scaler + (float)mask_block[i];
    if (max_input < x) {
      max_input = x;
    }
  }

  if (is_local_row) {
    for (int g = tid; g < global_num; g += blockSize) {
      int i = global_idx[g];
      if (i < col_start || i > col_end) {
        float x = input_block[i];
        x = x * scaler + (float)mask_block[i];
        if (max_input < x) {
          max_input = x;
        }
      }
    }
  }
  //__syncthreads();
  float max_block = BlockReduce(block_reduce_temp).Reduce(max_input, cub::Max());
  if (tid == 0) {
    max_shared = max_block;
  }
  __syncthreads();

  float sum_input = 0.f;
  // #pragma unroll 16
  for (int i = tid + col_start; i < col_end; i += blockSize) {
    float x = input_block[i];
    x = expf((x)*scaler + (float)mask_block[i] - max_shared);
    sum_input += x;
  }

  if (is_local_row) {
    for (int g = tid; g < global_num; g += blockSize) {
      int i = global_idx[g];
      if (i < col_start || i > col_end) {
        float x = input_block[i];
        x = expf((x)*scaler + (float)mask_block[i] - max_shared);
        sum_input += x;
      }
    }
  }

  //__syncthreads();
  float sum_block = BlockReduce(block_reduce_temp).Reduce(sum_input, cub::Sum());
  if (tid == 0) {
    sum_shared = sum_block;
  }
  __syncthreads();
  float recip_sum = 1.f / sum_shared;

  if (is_local_row) {
    // We only need to fill in zeros for blocks that will be used in the matrix multiplication
    // following the Softmax.
    //
    // For now zero-out only [row_index - 2*attention_window, row_index + 2*attention_window],
    // we can even be more agressive and reduce the zeroing out window size since
    // each row has entries in 3 blocks (3*attention_window size instead of 4*attention_window)
    int zero_start = row_index - 2 * attention_window;
    if (zero_start < 0) {
      zero_start = 0;
    }

    int zero_end = row_index + 2 * attention_window;
    if (zero_end > sequence_length) {
      zero_end = sequence_length;
    }

    for (int i = tid + zero_start; i < zero_end; i += blockSize) {
      output_block[i] = (T)(0.);
    }

    for (int g = tid; g < global_num; g += blockSize) {
      int i = global_idx[g];
      float x = input_block[i];
      x = expf((x)*scaler + (float)mask_block[i] - max_shared);
      output_block[i] = (T)(recip_sum * x);
    }
  }

  // #pragma unroll 16
  for (int i = tid + col_start; i < col_end; i += blockSize) {
    float x = input_block[i];
    x = expf((x)*scaler + (float)mask_block[i] - max_shared);
    output_block[i] = (T)(recip_sum * x);
  }
}

bool launchSoftmaxKernel(
    cudaStream_t stream,
    cublasHandle_t cublas,
    void* workspace,
    size_t softmax_workspace_size,
    const void* q,                // transposed Q with shape (B, N, S, H)
    const void* k,                // transposed K with shape (B, N, S, H)
    const void* v,                // transposed V with shape (B, N, S, H)
    const void* attention_mask,   // attention mask with shape (B, S), with value 0 not masked and -10000 masked.
    const void* global_q,         // Q for global tokens with shape (B, N, G, H)
    const void* global_k,         // K for global tokens with shape (B, N, S, H)
    const void* global_v,         // V for global tokens with shape (B, N, S, H)
    const int* global_attention,  // global attention with shape (B, S), with value 0 for local attention and 1 for global attention.
    void* output,                 // output with shape (B, N, S, H)
    float scaler,                 // scalar
    int batch_size,               // batch size
    int sequence_length,          // sequence length
    int num_heads,                // number of heads
    int head_size,                // hidden size per head
    int attention_window,         // one sided windows size
    int max_num_global,           // maximum number of global tokens (G) in all batches
    size_t element_size) {        // size of element: 2 for half, and 4 for float
  if (batch_size > MAX_LONGFORMER_BATCH_SIZE) {
    ORT_THROW("LongformerAttention CUDA operator does not support batch size > 128.");
  }

  bool is_fp16 = (element_size == 2);
  void* scratch1 = reinterpret_cast<char*>(workspace) + 3 * sizeof(int) * batch_size * sequence_length;
  void* scratch2 = reinterpret_cast<char*>(scratch1) + GetAttentionScratchSize(element_size, batch_size, num_heads, sequence_length, sequence_length);

  // Build index for global tokens
  int* global_idx = BuildGlobalIndex(stream, global_attention, batch_size, sequence_length, workspace, softmax_workspace_size);
  int* batch_global_num = global_idx + batch_size * sequence_length;

  int num_global[MAX_LONGFORMER_BATCH_SIZE] = {-1};
  if (!CUDA_CALL(cudaMemcpyAsync(&num_global[0], batch_global_num, batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream))) {
    return false;
  }

  // setup shared parameters for two strided batched matrix multiplies
  cudaDataType_t Atype;
  cudaDataType_t Btype;
  cudaDataType_t Ctype;
  cudaDataType_t resultType;
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

  __half one_fp16, zero_fp16;
  float one_fp32, zero_fp32;
  void *alpha, *beta_0, *beta_1;

  if (is_fp16) {
    one_fp16 = __float2half(1.f);
    zero_fp16 = __float2half(0.f);
    alpha = static_cast<void*>(&one_fp16);
    beta_0 = static_cast<void*>(&zero_fp16);
    beta_1 = static_cast<void*>(&one_fp16);
    Atype = CUDA_R_16F;
    Btype = CUDA_R_16F;
    Ctype = CUDA_R_16F;
    resultType = CUDA_R_16F;
    algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  } else {
    one_fp32 = 1.f;
    zero_fp32 = 0.f;
    alpha = static_cast<void*>(&one_fp32);
    beta_0 = static_cast<void*>(&zero_fp32);
    beta_1 = static_cast<void*>(&one_fp32);
    Atype = CUDA_R_32F;
    Btype = CUDA_R_32F;
    Ctype = CUDA_R_32F;
    resultType = CUDA_R_32F;
  }

  // Strided batch matrix multiply
  //    qk = q * k^T
  // Shapes: q and k = B x N x S x H, qk = B x N x S x S
  // Convert col-major to row-major by swapping q and k in Gemm

  // Local attention part
  // S x S is calculated using sliding block WxW (W is one sided window size) like the following:
  //   [W][W]
  //   [W][W][W]
  //      [W][W][W]
  //         [W][W]
  // The first and last rows have 2 blocks, and the remaining has 3 blocks per row.
  // The calculation are splited into 3 parts. Firstly, fill the middle rows,  then the first row and finally the last row.
  // The results are stored in scratch1.
  // TODO: Save space by not storing the whole matrix. Instead only allocate space for these blocks.
  
  int w = attention_window;
  int x_offset = num_heads * sequence_length * head_size;
  int y_offset = num_heads * sequence_length * sequence_length;
  int last_block = (sequence_length / w) - 1;
  int strideA = sequence_length * head_size;
  int strideB = sequence_length * head_size;
  int strideC = sequence_length * sequence_length;

  // When S == 2W, there is no middle rows of blocks:
  //   [W][W]
  //   [W][W]
  // We can use normal matrix multiplication in this case.
  if (sequence_length == 2 * w) {
    CHECK(cublasGemmStridedBatchedEx(cublas,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     sequence_length,
                                     sequence_length,
                                     head_size,
                                     alpha,
                                     k,
                                     Atype,
                                     head_size,
                                     sequence_length * head_size,
                                     q,
                                     Btype,
                                     head_size,
                                     sequence_length * head_size,
                                     beta_0,
                                     scratch1,
                                     Ctype,
                                     sequence_length,
                                     sequence_length * sequence_length,
                                     batch_size * num_heads,
                                     resultType,
                                     algo));
  } else {  // sequence_length > 2 * w
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < num_heads; ++j) {
        void* q_head = (char*)q + (i * x_offset + j * sequence_length * head_size + w * head_size) * element_size;
        void* k_head = (char*)k + (i * x_offset + j * sequence_length * head_size) * element_size;
        void* qk_head = (char*)scratch1 + (i * y_offset + j * sequence_length * sequence_length + w * sequence_length) * element_size;
        int count = (sequence_length - 2 * w) / w;
        CHECK(cublasGemmStridedBatchedEx(cublas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         3 * w,                    // m
                                         w,                        // n
                                         head_size,                // k
                                         alpha,                    // alpha
                                         k_head,                   // A
                                         Atype,                    // A type
                                         head_size,                // lda
                                         w * head_size,            // strideA
                                         q_head,                   // B
                                         Btype,                    // B type
                                         head_size,                // ldb
                                         w * head_size,            // strideB
                                         beta_0,                   // beta
                                         qk_head,                  // C
                                         Ctype,                    // C type
                                         sequence_length,          // ldc
                                         sequence_length * w + w,  // strideC
                                         count,                    // batch count
                                         resultType,
                                         algo));
      }
    }

    CHECK(cublasGemmStridedBatchedEx(cublas,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     2 * w,                   // m
                                     w,                       // n
                                     head_size,               // k
                                     alpha,                   // alpha
                                     k,                       // A
                                     Atype,                   // A type
                                     head_size,               // lda
                                     strideA,                 // strideA
                                     q,                       // B
                                     Btype,                   // B type
                                     head_size,               // ldb
                                     strideB,                 // strideB
                                     beta_0,                  // beta
                                     scratch1,                // C
                                     Ctype,                   // C type
                                     sequence_length,         // ldc
                                     strideC,                 // strideC
                                     batch_size * num_heads,  // batch count
                                     resultType,
                                     algo));

    void* q_head = (char*)q + (last_block * w * head_size) * element_size;
    void* k_head = (char*)k + ((last_block - 1) * w * head_size) * element_size;
    void* qk_head = (char*)scratch1 + (last_block * w * sequence_length + (last_block - 1) * w) * element_size;
    CHECK(cublasGemmStridedBatchedEx(cublas,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     2 * w,
                                     w,
                                     head_size,
                                     alpha,
                                     k_head,
                                     Atype,
                                     head_size,
                                     strideA,
                                     q_head,
                                     Btype,
                                     head_size,
                                     strideB,
                                     beta_0,
                                     qk_head,
                                     Ctype,
                                     sequence_length,
                                     strideC,
                                     batch_size * num_heads,
                                     resultType,
                                     algo));
  }

  // Global attention part
  for (int i = 0; i < batch_size; ++i) {
    if (num_global[i] > 0) {
      void* q_batch = (char*)q + (i * x_offset) * element_size;
      void* k_batch = (char*)k + (i * x_offset) * element_size;
      void* qk_batch = (char*)scratch1 + (i * y_offset) * element_size;
      // Local tokens attending global tokens
      CHECK(cublasGemmStridedBatchedEx(cublas,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       num_global[i],
                                       sequence_length,
                                       head_size,
                                       alpha,
                                       k_batch,
                                       Atype,
                                       head_size,
                                       strideA,
                                       q_batch,
                                       Btype,
                                       head_size,
                                       strideB,
                                       beta_0,
                                       qk_batch,
                                       Ctype,
                                       sequence_length,
                                       strideC,
                                       num_heads,
                                       resultType,
                                       algo));

      void* global_q_batch = (char*)global_q + (i * num_heads * max_num_global * head_size) * element_size;
      void* global_k_batch = (char*)global_k + (i * x_offset) * element_size;
      int strideB_global = max_num_global * head_size;

      // Global tokens attending everything
      // This GEMMs need to be last to make sure all global token entries are re-written.
      CHECK(cublasGemmStridedBatchedEx(cublas,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       sequence_length,
                                       num_global[i],
                                       head_size,
                                       alpha,
                                       global_k_batch,
                                       Atype,
                                       head_size,
                                       strideA,
                                       global_q_batch,
                                       Btype,
                                       head_size,
                                       strideB_global,
                                       beta_0,
                                       qk_batch,
                                       Ctype,
                                       sequence_length,
                                       strideC,
                                       num_heads,
                                       resultType,
                                       algo));
    }
  }

  int dim0 = sequence_length * num_heads;
  int dim1 = sequence_length;
  void* softmax_out = scratch2;

  const int blockSize = 64;
  const int gridSize = batch_size * num_heads * sequence_length;
  if (is_fp16) {
    LongformerSoftmaxKernel<__half, blockSize><<<gridSize, blockSize, 0, stream>>>(
        global_attention,
        global_idx,
        batch_global_num,
        static_cast<const __half*>(scratch1),
        static_cast<const __half*>(attention_mask),
        static_cast<__half*>(softmax_out), scaler, dim0, dim1, attention_window);
  } else {
    LongformerSoftmaxKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(
        global_attention,
        global_idx,
        batch_global_num,
        static_cast<const float*>(scratch1),
        static_cast<const float*>(attention_mask),
        static_cast<float*>(softmax_out), scaler, dim0, dim1, attention_window);
  }

  // Run the matrix multiply: output = softmax_out * v
  //   softmax_out: B x N x S x S
  //             v: B x N x S x H
  //      attn_out: B x N x S x H
  // Calculation uses full Gemm (S == 2W) or sliding blocks (S > 2W) in a way similar to local attention part.

  if (sequence_length == 2 * w) {
    // convert col-major to row-major by swapping softmax_out and v
    CHECK(cublasGemmStridedBatchedEx(cublas,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     head_size,
                                     sequence_length,
                                     sequence_length,
                                     alpha,
                                     v,
                                     Atype,
                                     head_size,
                                     sequence_length * head_size,
                                     softmax_out,
                                     Btype,
                                     sequence_length,
                                     sequence_length * sequence_length,
                                     beta_0,
                                     output,
                                     Ctype,
                                     head_size,
                                     sequence_length * head_size,
                                     batch_size * num_heads,
                                     resultType,
                                     algo));
  }
  else { // sequence_length > 2 * w
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < num_heads; ++j) {
        void* v_head = (char*)v + (i * x_offset + j * head_size * sequence_length) * element_size;
        void* prob_head = (char*)softmax_out + (i * y_offset + j * sequence_length * sequence_length + w * sequence_length) * element_size;
        void* out_head = (char*)output + (i * x_offset + j * head_size * sequence_length + w * head_size) * element_size;
        int count = (sequence_length - 2 * w) / w;
        CHECK(cublasGemmStridedBatchedEx(cublas,
                                         CUBLAS_OP_N,
                                         CUBLAS_OP_N,
                                         head_size,
                                         w,
                                         3 * w,
                                         alpha,
                                         v_head,
                                         Atype,
                                         head_size,
                                         w * head_size,
                                         prob_head,
                                         Btype,
                                         sequence_length,
                                         sequence_length * w + w,
                                         beta_0,
                                         out_head,
                                         Ctype,
                                         head_size,
                                         w * head_size,
                                         count,
                                         resultType,
                                         algo));
      }
    }

    CHECK(cublasGemmStridedBatchedEx(cublas,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     head_size,
                                     w,
                                     2 * w,
                                     alpha,
                                     v,
                                     Atype,
                                     head_size,
                                     sequence_length * head_size,
                                     softmax_out,
                                     Btype,
                                     sequence_length,
                                     sequence_length * sequence_length,
                                     beta_0,
                                     output,
                                     Ctype,
                                     head_size,
                                     sequence_length * head_size,
                                     batch_size * num_heads,
                                     resultType,
                                     algo));

    void* v_head = (char*)v + (last_block - 1) * w * head_size * element_size;
    void* prob_head = (char*)softmax_out + (sequence_length * last_block * w + (last_block - 1) * w) * element_size;
    void* out_head = (char*)output + last_block * w * head_size * element_size;

    CHECK(cublasGemmStridedBatchedEx(cublas,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     head_size,
                                     w,
                                     2 * w,
                                     alpha,
                                     v_head,
                                     Atype,
                                     head_size,
                                     sequence_length * head_size,
                                     prob_head,
                                     Btype,
                                     sequence_length,
                                     sequence_length * sequence_length,
                                     beta_0,
                                     out_head,
                                     Ctype,
                                     head_size,
                                     sequence_length * head_size,
                                     batch_size * num_heads,
                                     resultType,
                                     algo));
  }


  for (int i = 0; i < batch_size; ++i) {
    if (num_global[i] > 0) {
      int glob_longdim_mm = (last_block - 1) * w;

      void* v_head = (char*)v + (i * x_offset) * element_size;
      void* prob_head = (char*)softmax_out + (i * y_offset + 2 * w * sequence_length) * element_size;
      void* out_head = (char*)output + (i * x_offset + 2 * w * head_size) * element_size;

      CHECK(cublasGemmStridedBatchedEx(cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       head_size,
                                       glob_longdim_mm,
                                       num_global[i],
                                       alpha,
                                       v_head,
                                       Atype,
                                       head_size,
                                       sequence_length * head_size,
                                       prob_head,
                                       Btype,
                                       sequence_length,
                                       sequence_length * sequence_length,
                                       beta_1,
                                       out_head,
                                       Ctype,
                                       head_size,
                                       sequence_length * head_size,
                                       num_heads,
                                       resultType,
                                       algo));

      // Global tokens
      v_head = (char*)global_v + (i * x_offset) * element_size;
      prob_head = (char*)softmax_out + (i * y_offset) * element_size;
      out_head = (char*)output + (i * x_offset) * element_size;

      CHECK(cublasGemmStridedBatchedEx(cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       head_size,
                                       num_global[i],
                                       sequence_length,  // Re-write entries completely
                                       alpha,
                                       v_head,
                                       Atype,
                                       head_size,
                                       sequence_length * head_size,
                                       prob_head,
                                       Btype,
                                       sequence_length,
                                       sequence_length * sequence_length,
                                       beta_0,   // Use beta=0 to overwrite
                                       out_head, // Here assumes global tokens are at the beginning of sequence.
                                       Ctype,
                                       head_size,
                                       sequence_length * head_size,
                                       num_heads,
                                       resultType,
                                       algo));
    }
  }

  return true;
}

template <typename T>
bool LongformerQkvToContext(
    const cudaDeviceProp& prop, cublasHandle_t& cublas, cudaStream_t stream,
    const int batch_size, const int sequence_length, const int num_heads, const int head_size,
    const int window, const size_t element_size,
    const T* input, const T* attention_mask,
    const T* global_input, const int* global_attention, const int max_num_global,
    T* workspace,
    T* output) {
  size_t softmax_workspace_size = GetLongformerSoftmaxWorkspaceSize(element_size, batch_size, num_heads, sequence_length);
  T* qkv = reinterpret_cast<T*>((char*)workspace + softmax_workspace_size);

  // Input should be BxSx3xNxH => qkv: 3xBxNxSxH
  if (!LaunchTransQkv(stream, sequence_length, batch_size, head_size, num_heads, input, qkv)) {
    return false;
  }

  // Input 'global_input' should be BxSx3xNxH => global_qkv: 3xBxNxSxH
  T* global_qkv = qkv + 3 * batch_size * sequence_length * num_heads * head_size * element_size;

  // When there is no global token, no need to process global Q, K and V
  if (max_num_global > 0 && nullptr != global_input) {
    if (!LaunchTransQkv(stream, sequence_length, batch_size, head_size, num_heads, global_input, global_qkv)) {
      return false;
    }
  }

  // Now qkv has Q, K, V: each has size BxNxSxH
  const int elements = batch_size * num_heads * sequence_length * head_size;
  const T* q = qkv;
  const T* k = q + elements;
  const T* v = k + elements;

  const T* global_q = global_qkv;
  const T* global_k = global_q + elements;
  const T* global_v = global_k + elements;

  cublasSetStream(cublas, stream);
  CublasMathModeSetter helper(prop, cublas, CUBLAS_TENSOR_OP_MATH);

  // Q*K' are scaled by 1/sqrt(H)
  const float rsqrt_head_size = 1.f / sqrt(static_cast<float>(head_size));

  cudaDeviceSynchronize();

  T* temp_output = qkv;  // Q will be overwritten
  if (!launchSoftmaxKernel(
          stream,
          cublas,
          workspace,
          softmax_workspace_size,
          q,                 // Transposed Q with shape B x N x S x H
          k,                 // Transposed K with shape B x N x S x H
          v,                 // Transposed V with shape B x N x S x H
          attention_mask,    // Attention mask flags with shape B x S
          global_q,          // Transposed global Q with shape B x N x G x H
          global_k,          // Transposed global K with shape B x N x S x H
          global_v,          // Transposed global V with shape B x N x S x H
          global_attention,  // Global attention flags with shape B x S
          temp_output,       // Output with shape B x N x S x H
          rsqrt_head_size,   // Scaler
          batch_size,        // Batch size
          sequence_length,   // Sequence length
          num_heads,         // Number of attention heads
          head_size,         // Hidden size per head
          window,            // Half (one-sided) windows size
          max_num_global,    // Maximum number of global tokens (G)
          element_size)) {
    return false;
  }

  // The temp_output is BxNxSxH, transpose it to final output BxSxNxH
  return LaunchTransCtx(stream, sequence_length, batch_size, head_size, num_heads, temp_output, output);
}

bool LaunchLongformerAttentionKernel(
    const cudaDeviceProp& prop,
    cudaStream_t stream,
    const void* input,
    const void* attention_mask,
    const void* global_input,
    const int* global_attention,
    void* output,
    int batch_size,
    int sequence_length,
    int num_heads,
    int head_size,
    int window,
    int max_num_global,
    void* workspace,
    cublasHandle_t& cublas,
    const size_t element_size) {
  if (element_size == 2) {
    return LongformerQkvToContext(prop, cublas, stream,
                                  batch_size, sequence_length, num_heads, head_size, window, element_size,
                                  reinterpret_cast<const half*>(input),
                                  reinterpret_cast<const half*>(attention_mask),
                                  reinterpret_cast<const half*>(global_input),
                                  global_attention,
                                  max_num_global,
                                  reinterpret_cast<half*>(workspace),
                                  reinterpret_cast<half*>(output));
  } else {
    return LongformerQkvToContext(prop, cublas, stream,
                                  batch_size, sequence_length, num_heads, head_size, window, element_size,
                                  reinterpret_cast<const float*>(input),
                                  reinterpret_cast<const float*>(attention_mask),
                                  reinterpret_cast<const float*>(global_input),
                                  global_attention,
                                  max_num_global,
                                  reinterpret_cast<float*>(workspace),
                                  reinterpret_cast<float*>(output));
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
