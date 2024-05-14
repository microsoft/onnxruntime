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
// (2) Maximum number of global tokens <= one-sided attention window

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <library_types.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/bert/add_bias_transpose.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/longformer_attention_softmax.h"
#include "contrib_ops/cuda/bert/longformer_attention_impl.h"

using namespace onnxruntime::cuda;
using namespace cub;

#define CHECK(expr) CUBLAS_RETURN_IF_ERROR(expr)
#define CHECK_CUDA(expr) CUDA_RETURN_IF_ERROR(expr)

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Denote: batch size (B), sequence length (S), number of heads (N), dimension per head (H), maximum global tokens (G)
//
// Workspace layout (default data type T is float or half):
//   [SoftmaxSpace] [Q:BxNxSxH] [K:BxNxSxH] [V:BxNxSxH] [Global_Q:BxNxSxH] [Global_K:BxNxSxH] [Global_V:BxNxSxH]
// where Global_Q, Global_K and Global_V are optional. They are not allocated when there is no global token.
//
// SoftmaxSpace layout is the following when compact memory is enabled:
//    [scratch1: (5S-3W)*W*N*B] [scratch2: size_t 15]
// Scratch1 has 5 buffers for local and global attention calculation.
// Scratch2 has 5 input/output pointers, 5 buffer sizes and 5 strides related to scratch1.
//
// SoftmaxSpace layout is the following When compact memory is disabled:
//    [scratch1: BxNxSxS] [scratch2: BxNxSxS]

static size_t Align(size_t a) {
  const size_t alignment = 128;  // Align on a 16-byte boundary to avoid "misaligned address" error.
  return CeilDiv(a, alignment) * alignment;
}

size_t GetScratch1Size(size_t element_size, size_t batch_size, size_t num_heads, size_t sequence_length, size_t window) {
  size_t bytes = (5 * sequence_length - 3 * window) * window * num_heads * batch_size * element_size;
  return Align(bytes);
}

constexpr size_t GetScratch2Size() {
  return 5 * sizeof(void*) + 10 * sizeof(size_t);
}

size_t GetLongformerSoftmaxWorkspaceSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t sequence_length,
    size_t window,
    bool disable_compact_memory) {
  if (!disable_compact_memory) {
    size_t scratch1_size = GetScratch1Size(element_size, batch_size, num_heads, sequence_length, window);
    size_t scratch2_size = GetScratch2Size();
    return Align(scratch1_size + scratch2_size);
  } else {
    return 2 * GetAttentionScratchSize(element_size, batch_size, num_heads, sequence_length, sequence_length);
  }
}

size_t GetLongformerAttentionWorkspaceSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t head_size,
    size_t sequence_length,
    size_t max_num_global,
    size_t window,
    bool disable_compact_memory) {
  size_t softmax_size = GetLongformerSoftmaxWorkspaceSize(element_size,
                                                          batch_size,
                                                          num_heads,
                                                          sequence_length,
                                                          window,
                                                          disable_compact_memory);
  size_t qkv_size = static_cast<size_t>(3) * batch_size * sequence_length * num_heads * head_size * element_size;
  size_t global_qkv_size = max_num_global > 0 ? qkv_size : 0;
  return softmax_size + qkv_size + global_qkv_size;
}

// Size of buffer of pinned memory in CPU. The buffer is used to copy memory between CPU and GPU.
// The buffer includes two parts: [global_count (copy of batch_global_num): int Bx1] [copy of scratch2]
size_t GetPinnedBufferSize(size_t batch_size) {
  return sizeof(int) * batch_size + GetScratch2Size();
}

// Softmax kernel for compact format
template <typename T, int blockSize>
__launch_bounds__(blockSize)
    __global__ void LongformerSoftmaxKernel(const int* global_attention,
                                            const int* global_index,
                                            const int* batch_global_num,
                                            void* buffer_pointers,
                                            const T* attention_mask,
                                            float scaler,
                                            int sequence_length,
                                            int num_heads,
                                            int window) {
  typedef cub::BlockReduce<float, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage block_reduce_temp;

  int tid = threadIdx.x;
  const int batch_index = blockIdx.x / (sequence_length * num_heads);
  const int row_index = blockIdx.x % sequence_length;
  const int head_index = (blockIdx.x / sequence_length) % num_heads;

  // Adjust the pointers for the batch
  const T* mask_block = attention_mask + sequence_length * batch_index;
  const int* global_index_block = global_index + sequence_length * batch_index;
  const int global_num = batch_global_num[batch_index];
  size_t* p_inputs = reinterpret_cast<size_t*>(buffer_pointers);
  size_t* p_outputs = reinterpret_cast<size_t*>(buffer_pointers);
  size_t* input_sizes = reinterpret_cast<size_t*>(buffer_pointers) + 5;
  size_t* input_strides = reinterpret_cast<size_t*>(buffer_pointers) + 10;
  const T* inputs[5];
  T* outputs[5];
  for (int i = 0; i < 5; ++i) {
    inputs[i] = reinterpret_cast<T*>(p_inputs[i]) + batch_index * num_heads * input_sizes[i];
    outputs[i] = reinterpret_cast<T*>(p_outputs[i]) + batch_index * num_heads * input_sizes[i];
  }

  // Local attention token
  int col_start = 0;
  int col_end = sequence_length;
  bool is_local_row = (global_attention[batch_index * sequence_length + row_index] == static_cast<int>(0));
  if (is_local_row) {
    col_start = row_index - window;
    if (col_start < 0) {
      col_start = 0;
    }

    col_end = row_index + window + 1;
    if (col_end > sequence_length) {
      col_end = sequence_length;
    }
  }

  // If mask is set then set everything to zero to match huggingface transformers implementation
  if ((float)mask_block[row_index] != 0.f) {
    if (is_local_row) {
      T* output_block = nullptr;
      T* output_global = nullptr;
      int local_offset = row_index % window;
      int local_start = 0;
      int local_end = 3 * window;
      if (row_index < window) {
        local_start = 0;
        local_end = 2 * window;
        output_block = outputs[0] + row_index * input_strides[0] + head_index * input_sizes[0];
      } else if (row_index < sequence_length - window) {
        output_block = outputs[1] + (row_index - window) * input_strides[1] + head_index * input_sizes[1];
      } else {
        local_start = 0;
        local_end = 2 * window;
        output_block = outputs[2] + local_offset * input_strides[2] + head_index * input_sizes[2];
      }

      for (int i = local_start + tid; i < local_end; i += blockSize) {
        output_block[i] = 0;
      }

      if ((row_index - 2 * window) >= 0) {
        output_global = outputs[3] + (row_index - window) * input_strides[3] + head_index * input_sizes[3];
      }

      if (output_global != nullptr) {
        for (int i = tid; i < global_num; i += blockSize) {
          output_global[i] = 0;
        }
      }

    } else {
      T* output_block = outputs[4];
      for (int i = tid; i < sequence_length; i += blockSize)
        output_block[i] = 0;
    }
    return;
  }

  float sum_input = 0.;
  __shared__ float sum_shared;

  // Calculate max input
  float max_input = -CUDART_INF_F;
  __shared__ float max_shared;

  if (is_local_row) {
    const T* input_block = nullptr;
    T* output_block = nullptr;
    T* output_global = nullptr;
    int local_offset = row_index % window;
    int local_start = local_offset;
    int local_end = local_start + 2 * window + 1;
    int zero_start = 0;
    int zero_end = 3 * window;
    if (row_index < window) {
      local_start = 0;
      local_end = local_offset + window + 1;
      zero_end = 2 * window;

      input_block = inputs[0] + row_index * input_strides[0] + head_index * input_sizes[0];
      output_block = outputs[0] + row_index * input_strides[0] + head_index * input_sizes[0];
    } else if (row_index < sequence_length - window) {
      input_block = inputs[1] + (row_index - window) * input_strides[1] + head_index * input_sizes[1];
      output_block = outputs[1] + (row_index - window) * input_strides[1] + head_index * input_sizes[1];
    } else {
      local_start = local_offset;
      local_end = 2 * window;
      zero_end = 2 * window;

      input_block = inputs[2] + local_offset * input_strides[2] + head_index * input_sizes[2];
      output_block = outputs[2] + local_offset * input_strides[2] + head_index * input_sizes[2];
    }

    const T* input_global = nullptr;
    int local_global = row_index - window;
    if (local_global > global_num) {
      local_global = global_num;
    }
    if (local_global > 0) {
      input_global = inputs[3] + (row_index - window) * input_strides[3] + head_index * input_sizes[3];
    }

    if (row_index < window) {
      output_global = (T*)outputs[0] + row_index * input_strides[0] + head_index * input_sizes[0];
    } else if (row_index < 2 * window) {
      output_global = outputs[1] + (row_index - window) * input_strides[1] + head_index * input_sizes[1];
    } else {
      output_global = outputs[3] + (row_index - window) * input_strides[3] + head_index * input_sizes[3];
    }

    for (int i = local_start + tid, j = col_start + tid; i < local_end; i += blockSize, j += blockSize) {
      float x = input_block[i];
      x = x * scaler + (float)mask_block[j];
      if (max_input < x)
        max_input = x;
    }

    if (input_global != nullptr) {
      for (int i = tid; i < local_global; i += blockSize) {
        float x = input_global[global_index_block[i]];
        x = x * scaler + (float)mask_block[global_index_block[i]];
        if (max_input < x)
          max_input = x;
      }
    }

    float max_block = BlockReduce(block_reduce_temp).Reduce(max_input, cub::Max());
    if (tid == 0) {
      max_shared = max_block;
    }
    __syncthreads();

    for (int i = local_start + tid, j = col_start + tid; i < local_end; i += blockSize, j += blockSize) {
      float x = input_block[i];
      x = expf((x)*scaler + (float)mask_block[j] - max_shared);
      sum_input += x;
    }

    if (input_global != nullptr) {
      for (int i = tid, j = col_start + tid; i < local_global; i += blockSize, j += blockSize) {
        float x = input_global[global_index_block[i]];
        x = expf((x)*scaler + (float)mask_block[j] - max_shared);
        sum_input += x;
      }
    }

    float sum_block = BlockReduce(block_reduce_temp).Reduce(sum_input, cub::Sum());
    if (tid == 0) {
      sum_shared = sum_block;
    }
    __syncthreads();
    float recip_sum = 1.f / sum_shared;

    for (int i = tid + zero_start; i < local_start; i += blockSize) {
      output_block[i] = (T)(0.);
    }

    for (int i = tid + local_end; i < zero_end; i += blockSize) {
      output_block[i] = (T)(0.);
    }

    __syncthreads();

    for (int i = local_start + tid, j = col_start + tid; i < local_end; i += blockSize, j += blockSize) {
      float x = input_block[i];
      x = expf((x)*scaler + (float)mask_block[j] - max_shared);
      output_block[i] = (T)(recip_sum * x);
    }

    if (input_global != nullptr) {
      for (int i = tid; i < local_global; i += blockSize) {
        float x = input_global[global_index_block[i]];
        x = expf((x)*scaler + (float)mask_block[global_index_block[i]] - max_shared);
        output_global[i] = (T)(recip_sum * x);
      }
    }
  } else {
    // Global tokens
    const T* input_block = inputs[4] + row_index * input_strides[4] + head_index * input_sizes[4];
    T* output_block = outputs[4] + row_index * input_strides[4] + head_index * input_sizes[4];

    for (int i = tid; i < sequence_length; i += blockSize) {
      float x = input_block[i];
      x = x * scaler + (float)mask_block[i];
      if (max_input < x)
        max_input = x;
    }

    float max_block = BlockReduce(block_reduce_temp).Reduce(max_input, cub::Max());
    if (tid == 0) {
      max_shared = max_block;
    }
    __syncthreads();

    for (int i = tid; i < sequence_length; i += blockSize) {
      float x = input_block[i];
      x = expf((x)*scaler + (float)mask_block[i] - max_shared);
      sum_input += x;
    }

    float sum_block = BlockReduce(block_reduce_temp).Reduce(sum_input, cub::Sum());
    if (tid == 0) {
      sum_shared = sum_block;
    }
    __syncthreads();
    float recip_sum = 1.f / sum_shared;

    for (int i = tid; i < sequence_length; i += blockSize) {
      float x = input_block[i];
      x = expf((x)*scaler + (float)mask_block[i] - max_shared);
      output_block[i] = (T)(recip_sum * x);
    }
  }
}

Status LaunchLongformerSoftmaxKernel(
    cudaStream_t stream,
    cublasHandle_t cublas,
    void* workspace,
    const void* q,                // transposed Q with shape (B, N, S, H)
    const void* k,                // transposed K with shape (B, N, S, H)
    const void* v,                // transposed V with shape (B, N, S, H)
    const void* attention_mask,   // attention mask with shape (B, S), with value 0 not masked and value of mask_filter_value.
    int max_num_global,           // maximum number of global tokens (G)
    const bool compact_global_q,  // whether global_q has shape (B, N, G, H) instead of (B, N, S, H)
    const void* global_q,         // Q for global tokens with shape (B, N, S, H).
    const void* global_k,         // K for global tokens with shape (B, N, S, H)
    const void* global_v,         // V for global tokens with shape (B, N, S, H)
    const int* global_attention,  // global attention flags with shape (B, S), with value 0 for local and 1 for global.
    const int* global_index,      // Global index with shape (B, S)
    const int* batch_global_num,  // Number of global tokens per batch with shape (B, 1)
    void* pinned_buffer,          // Pinned memory in CPU with 2 parts: global tokens per batch, and data for scratch2
    void* output,                 // output with shape (B, N, S, H)
    float scaler,                 // scalar
    int batch_size,               // batch size
    int sequence_length,          // sequence length
    int num_heads,                // number of heads
    int head_size,                // hidden size per head
    int window,                   // one sided window size
    size_t element_size) {        // size of element: 2 for half, and 4 for float
  const int* global_count = reinterpret_cast<const int*>(pinned_buffer);

  bool is_fp16 = (element_size == 2);
  char* scratch1 = reinterpret_cast<char*>(workspace);
  char* scratch2 = scratch1 + GetScratch1Size(element_size, batch_size, num_heads, sequence_length, window);

  // Setup shared parameters for two strided batched matrix multiplies
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
  size_t elements_per_batch = num_heads * sequence_length * head_size;
  int stride_per_head = sequence_length * head_size;  // stride for Q, K, V and output

  // Local attention part
  // S x S is calculated using sliding block WxW (W is one sided window size) like the following:
  //   [W][W]
  //   [W][W][W]
  //      [W][W][W]
  //         [W][W]
  // The first and last rows have 2 blocks per row, and the remaining has 3 blocks per row.
  // The calculation are splited into 3 parts: the first row, middle rows and finally the last row.
  // To save space, we do not store the whole matrix. Instead, we only allocate space for these blocks.
  //
  // For global attention part, we have two assumptions:
  // (1) Global tokens are at the beginging of sequence
  // (2) Number of global tokens <= attention window
  //
  // The results are stored in scratch1 buffer:
  //   Number of elements for local attention are (3*S/W-2)*W*W*N*B, or (3S-2W)*W*N*B
  //   Number of elements for local attends to global are (S-W)*W*N*B
  //   Number of elements for global attends to everything are S*W*N*B
  // Total elements (FP16 or FP32) are (5S-3W)*W*N*B

  const int w = window;
  const int middle_count = (sequence_length - 2 * w) / w;
  int last_block = (sequence_length / w) - 1;

  // Determine the non-zero block dimensions and pointers

  // Buffer size per head for a single batch
  size_t buffer_sizes[5] = {
      static_cast<size_t>(w * w * 2),                  // first row of blocks has 2 WxW blocks
      static_cast<size_t>(w * w * middle_count * 3),   // middle rows of blocks have 3 WxW blocks per row
      static_cast<size_t>(w * w * 2),                  // last row of blocks has 2 WxW blocks
      static_cast<size_t>(w * (sequence_length - w)),  // local attends to global: global tokens <= window size
      static_cast<size_t>(w * sequence_length)};       // global attends to everything.

  size_t buffer_strides[5] = {
      static_cast<size_t>(w * 2),
      static_cast<size_t>(w * 3),
      static_cast<size_t>(w * 2),
      static_cast<size_t>(w),  // number of global tokens <= window size
      static_cast<size_t>(sequence_length)};

  void* buffer_pointers[5];

  char* current_pointer = scratch1;
  for (int i = 0; i < 5; ++i) {
    buffer_pointers[i] = reinterpret_cast<void*>(current_pointer);
    current_pointer += buffer_sizes[i] * num_heads * batch_size * element_size;
  }

  // Copy to a continues buffer first so that we only need call cudaMemcpyAsync once

  char* temp_buffer = reinterpret_cast<char*>(pinned_buffer) + sizeof(int) * batch_size;
  memcpy(temp_buffer, &buffer_pointers[0], 5 * sizeof(void*));
  memcpy(temp_buffer + 5 * sizeof(void*), &buffer_sizes[0], 5 * sizeof(size_t));
  memcpy(temp_buffer + 5 * sizeof(void*) + 5 * sizeof(size_t), &buffer_strides[0], 5 * sizeof(size_t));
  CHECK_CUDA(cudaMemcpyAsync(scratch2, temp_buffer, GetScratch2Size(), cudaMemcpyHostToDevice, stream));

  // Local attention part
  {
    // local attention per head - head
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
                                     stride_per_head,         // strideA
                                     q,                       // B
                                     Btype,                   // B type
                                     head_size,               // ldb
                                     stride_per_head,         // strideB
                                     beta_0,                  // beta
                                     buffer_pointers[0],      // C
                                     Ctype,                   // C type
                                     2 * w,                   // ldc
                                     buffer_sizes[0],         // strideC
                                     batch_size * num_heads,  // batch count
                                     resultType,
                                     algo));

    // local attention per head - middle
    if (middle_count > 0) {
      for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_heads; ++j) {
          const void* q_head = reinterpret_cast<const char*>(q) +
                               (i * elements_per_batch + (j * sequence_length + w) * head_size) * element_size;
          const void* k_head = reinterpret_cast<const char*>(k) +
                               (i * elements_per_batch + j * sequence_length * head_size) * element_size;
          void* qk_head = reinterpret_cast<char*>(buffer_pointers[1]) +
                          static_cast<size_t>(i * num_heads + j) * buffer_sizes[1] * element_size;
          CHECK(cublasGemmStridedBatchedEx(cublas,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_N,
                                           3 * w,          // m
                                           w,              // n
                                           head_size,      // k
                                           alpha,          // alpha
                                           k_head,         // A
                                           Atype,          // A type
                                           head_size,      // lda
                                           w * head_size,  // strideA
                                           q_head,         // B
                                           Btype,          // B type
                                           head_size,      // ldb
                                           w * head_size,  // strideB
                                           beta_0,         // beta
                                           qk_head,        // C
                                           Ctype,          // C type
                                           3 * w,          // ldc
                                           3 * w * w,      // strideC
                                           middle_count,   // batch count
                                           resultType,
                                           algo));
        }
      }
    }

    // local attention per head - tail
    const void* q_head = reinterpret_cast<const char*>(q) + (last_block * w * head_size) * element_size;
    const void* k_head = reinterpret_cast<const char*>(k) + ((last_block - 1) * w * head_size) * element_size;

    CHECK(cublasGemmStridedBatchedEx(cublas,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     2 * w,                   // m
                                     w,                       // n
                                     head_size,               // k
                                     alpha,                   // alpha
                                     k_head,                  // A
                                     Atype,                   // A type
                                     head_size,               // lda
                                     stride_per_head,         // strideA
                                     q_head,                  // B
                                     Btype,                   // B type
                                     head_size,               // ldb
                                     stride_per_head,         // strideB
                                     beta_0,                  // beta
                                     buffer_pointers[2],      // C
                                     Ctype,                   // C type
                                     2 * w,                   // ldc
                                     buffer_sizes[2],         // strideC
                                     batch_size * num_heads,  // batch count
                                     resultType,
                                     algo));
  }

  // Global attention part
  for (int i = 0; i < batch_size; ++i) {
    if (global_count[i] > 0) {
      const void* q_batch = reinterpret_cast<const char*>(q) + (i * elements_per_batch + w * head_size) * element_size;
      const void* k_batch = reinterpret_cast<const char*>(k) + (i * elements_per_batch) * element_size;
      void* qk_batch = reinterpret_cast<char*>(buffer_pointers[3]) + (i * buffer_sizes[3]) * num_heads * element_size;

      // Local tokens attending global tokens
      CHECK(cublasGemmStridedBatchedEx(cublas,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       global_count[i],      // m
                                       sequence_length - w,  // n
                                       head_size,            // k
                                       alpha,                // alpha
                                       k_batch,              // A
                                       Atype,                // A type
                                       head_size,            // lda
                                       stride_per_head,      // strideA
                                       q_batch,              // B
                                       Btype,                // B type
                                       head_size,            // ldb
                                       stride_per_head,      // strideB
                                       beta_0,               // beta
                                       qk_batch,             // C
                                       Ctype,                // C type
                                       w,                    // ldc
                                       buffer_sizes[3],      // strideC
                                       num_heads,            // batch count
                                       resultType,
                                       algo));

      const size_t global_q_per_batch = compact_global_q ? num_heads * max_num_global * head_size : elements_per_batch;
      const int global_q_stride = (compact_global_q ? max_num_global * head_size : stride_per_head);
      const void* global_q_batch = reinterpret_cast<const char*>(global_q) + (i * global_q_per_batch) * element_size;
      const void* global_k_batch = reinterpret_cast<const char*>(global_k) + (i * elements_per_batch) * element_size;
      qk_batch = reinterpret_cast<char*>(buffer_pointers[4]) + (i * buffer_sizes[4] * num_heads) * element_size;

      // Global tokens attending everything
      // This GEMMs need to be last to make sure all global token entries are re-written.
      CHECK(cublasGemmStridedBatchedEx(cublas,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       sequence_length,  // m
                                       global_count[i],  // n
                                       head_size,        // k
                                       alpha,            // alpha
                                       global_k_batch,   // A
                                       Atype,            // A type
                                       head_size,        // lda
                                       stride_per_head,  // strideA
                                       global_q_batch,   // B
                                       Btype,            // B type
                                       head_size,        // ldb
                                       global_q_stride,  // strideB.
                                       beta_0,           // beta
                                       qk_batch,         // C
                                       Ctype,            // C type
                                       sequence_length,  // ldc
                                       buffer_sizes[4],  // strideC
                                       num_heads,        // batch count
                                       resultType,
                                       algo));
    }
  }

  const int blockSize = 64;
  const int gridSize = batch_size * num_heads * sequence_length;
  if (is_fp16) {
    LongformerSoftmaxKernel<__half, blockSize><<<gridSize, blockSize, 0, stream>>>(
        global_attention,
        global_index,
        batch_global_num,
        scratch2,
        static_cast<const __half*>(attention_mask),
        scaler, sequence_length, num_heads, window);
  } else {
    LongformerSoftmaxKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(
        global_attention,
        global_index,
        batch_global_num,
        scratch2,
        static_cast<const float*>(attention_mask),
        scaler, sequence_length, num_heads, window);
  }

  // local values attending the softmax score.
  {
    // local attention per head - head
    CHECK(cublasGemmStridedBatchedEx(cublas,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     head_size,                            // m
                                     w,                                    // n
                                     2 * w,                                // k
                                     alpha,                                // alpha
                                     v,                                    // A
                                     Atype,                                // A type
                                     head_size,                            // lda
                                     stride_per_head,                      // strideA
                                     buffer_pointers[0],                   // B
                                     Btype,                                // B type
                                     static_cast<int>(buffer_strides[0]),  // ldb
                                     buffer_sizes[0],                      // strideB
                                     beta_0,                               // beta
                                     output,                               // C
                                     Ctype,                                // C type
                                     head_size,                            // ldc
                                     stride_per_head,                      // strideC
                                     batch_size * num_heads,               // batch count
                                     resultType,
                                     algo));

    // local attention per head - middle
    if (middle_count > 0) {
      for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_heads; ++j) {
          const void* v_head = reinterpret_cast<const char*>(v) +
                               (i * elements_per_batch + j * head_size * sequence_length) * element_size;
          const void* prob_head = reinterpret_cast<const char*>(buffer_pointers[1]) +
                                  (i * num_heads + j) * buffer_sizes[1] * element_size;
          void* out_head = reinterpret_cast<char*>(output) +
                           (i * elements_per_batch + j * head_size * sequence_length + w * head_size) * element_size;
          CHECK(cublasGemmStridedBatchedEx(cublas,
                                           CUBLAS_OP_N,
                                           CUBLAS_OP_N,
                                           head_size,                            // m
                                           w,                                    // n
                                           3 * w,                                // k
                                           alpha,                                // alpha
                                           v_head,                               // A
                                           Atype,                                // A type
                                           head_size,                            // lda
                                           w * head_size,                        // strideA
                                           prob_head,                            // B
                                           Btype,                                // B type
                                           static_cast<int>(buffer_strides[1]),  // ldb
                                           3 * w * w,                            // strideB
                                           beta_0,                               // beta
                                           out_head,                             // C
                                           Ctype,                                // C type
                                           head_size,                            // ldc
                                           w * head_size,                        // strideC
                                           middle_count,                         // batch count
                                           resultType,
                                           algo));
        }
      }
    }

    // local attention per head - tail
    const void* v_head = reinterpret_cast<const char*>(v) + (last_block - 1) * w * head_size * element_size;
    void* out_head = reinterpret_cast<char*>(output) + last_block * w * head_size * element_size;

    CHECK(cublasGemmStridedBatchedEx(cublas,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     head_size,                            // m
                                     w,                                    // n
                                     2 * w,                                // k
                                     alpha,                                // alpha
                                     v_head,                               // A
                                     Atype,                                // A type
                                     head_size,                            // lda
                                     stride_per_head,                      // strideA
                                     buffer_pointers[2],                   // B
                                     Btype,                                // B type
                                     static_cast<int>(buffer_strides[2]),  // ldb
                                     buffer_sizes[2],                      // strideB
                                     beta_0,                               // beta
                                     out_head,                             // C
                                     Ctype,                                // C type
                                     head_size,                            // ldc
                                     stride_per_head,                      // strideC
                                     batch_size * num_heads,               // batch count
                                     resultType,
                                     algo));
  }

  // global attention part
  for (int i = 0; i < batch_size; ++i) {
    if (global_count[i] > 0) {
      // Local tokens attending global tokens
      const void* v_head = reinterpret_cast<const char*>(v) + (i * elements_per_batch) * element_size;
      const void* prob_head = reinterpret_cast<const char*>(buffer_pointers[3]) +
                              (i * buffer_sizes[3] * num_heads + w * buffer_strides[3]) * element_size;
      void* out_head = reinterpret_cast<char*>(output) + (i * elements_per_batch + 2 * w * head_size) * element_size;

      CHECK(cublasGemmStridedBatchedEx(cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       head_size,                            // m
                                       sequence_length - 2 * w,              // n
                                       global_count[i],                      // k
                                       alpha,                                // alpha
                                       v_head,                               // A
                                       Atype,                                // A type
                                       head_size,                            // lda
                                       stride_per_head,                      // strideA
                                       prob_head,                            // B
                                       Btype,                                // B type
                                       static_cast<int>(buffer_strides[3]),  // ldb
                                       buffer_sizes[3],                      // strideB
                                       beta_1,                               // beta
                                       out_head,                             // C
                                       Ctype,                                // C type
                                       head_size,                            // ldc
                                       stride_per_head,                      // strideC
                                       num_heads,                            // batch count
                                       resultType,
                                       algo));

      // Global tokens attending everything
      v_head = reinterpret_cast<const char*>(global_v) + (i * elements_per_batch) * element_size;
      prob_head = reinterpret_cast<const char*>(buffer_pointers[4]) + (i * buffer_sizes[4] * num_heads) * element_size;
      out_head = reinterpret_cast<char*>(output) + (i * elements_per_batch) * element_size;

      CHECK(cublasGemmStridedBatchedEx(cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       head_size,                            // m
                                       global_count[i],                      // n
                                       sequence_length,                      // k: re-write entries completely
                                       alpha,                                // alpha
                                       v_head,                               // A
                                       Atype,                                // A type
                                       head_size,                            // lda
                                       stride_per_head,                      // strideA
                                       prob_head,                            // B
                                       Btype,                                // B type
                                       static_cast<int>(buffer_strides[4]),  // ldb
                                       buffer_sizes[4],                      // strideB
                                       beta_0,                               // beta: overwrite
                                       out_head,                             // C: assumes global tokens at the beginning of sequence
                                       Ctype,                                // C type
                                       head_size,                            // ldc
                                       stride_per_head,                      // strideC
                                       num_heads,                            // batch count
                                       resultType,
                                       algo));
    }
  }

  return Status::OK();
}

template <typename T>
Status LongformerQkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t cublas,
    cudaStream_t stream,
    const int batch_size,       // batch size
    const int sequence_length,  // sequence length
    const int num_heads,        // number of attention heads
    const int head_size,        // hidden size per head
    const int window,           // Half (one-sided) window size
    const size_t element_size,
    const T* input,               // input for transpose
    const T* bias,                // bias to add to transposed input
    const T* attention_mask,      // attention mask with shape (B, S), with value 0.0 not masked, and -10000.0 or torch.finfo(dtype).min masked.
    const T* global_input,        // global input for transpose
    const T* global_bias,         // bias to add to transposed global input
    const int* global_attention,  // global attention flags with shape (B, S), with value 0 for local and 1 for global.
    const int* global_index,      // Global index with shape (B, S)
    const int* batch_global_num,  // Number of global tokens per batch with shape (B, 1)
    const int max_num_global,     // Maximum number of global tokens (G)
    void* pinned_buffer,          // Pinned memory in CPU. Number of global tokens per batch with shape (B, 1)
    T* workspace,                 // Softmax space
    T* output,                    // output
    size_t softmax_workspace_size,
    bool disable_compact_memory,
    bool use_merged_qkv_weights,
    bool use_half4) {
  T* qkv = reinterpret_cast<T*>(reinterpret_cast<char*>(workspace) + softmax_workspace_size);

  // Number of elements in Q, K, V, Global_Q, Global_K or Global_V are same: BxNxSxH
  const int elements = batch_size * num_heads * sequence_length * head_size;

  const int max_threads_per_block(device_prop.maxThreadsPerBlock);

  const int format = static_cast<int>(use_merged_qkv_weights);
  bool compact_global_q = false;
  // The order of qkv space:
  //  Q, K, V, Global_K, Global_V, Global_Q (format 0)
  //  Q, K, V, Global_Q, Global_K, Global_V (format 1)
  // Assume Q, K and V has same hidden size
  if (format == 1 || max_num_global == 0 || nullptr == global_input) {
    if (bias == nullptr) {
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 3, sequence_length, batch_size, head_size, num_heads,
                                         max_threads_per_block, false, input, qkv));
    } else {
      LaunchAddBiasTranspose(stream, 3, format, max_threads_per_block, batch_size,
                             sequence_length, num_heads, head_size,
                             input, bias, qkv,
                             use_half4, head_size);
    }

    if (max_num_global > 0 && nullptr != global_input) {
      if (global_bias == nullptr) {
        ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 3, sequence_length, batch_size, head_size, num_heads,
                                           max_threads_per_block, false, global_input, qkv + 3 * elements));
      } else {
        LaunchAddBiasTranspose(stream, 3, format, max_threads_per_block, batch_size,
                               sequence_length, num_heads, head_size,
                               global_input, global_bias, qkv + 3 * elements,
                               use_half4, head_size);
      }
    }
  } else {
    LaunchAddBiasTranspose(stream, 5, format, max_threads_per_block, batch_size,
                           sequence_length, num_heads, head_size,
                           input, bias, qkv,
                           use_half4, head_size);

    compact_global_q = (disable_compact_memory == false);
    LaunchAddBiasTranspose(stream, 1, format, max_threads_per_block, batch_size,
                           compact_global_q ? max_num_global : sequence_length, num_heads, head_size,
                           global_input + 2 * elements, global_bias, qkv + 5 * elements,
                           use_half4, head_size);
  }
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  // Transposed Q, K, V with shape (B, N, S, H)
  const T* q = qkv;
  const T* k = q + elements;
  const T* v = k + elements;

  // Transposed global Q, K, V with shape (B, N, S, H).
  // When compact_global_q is true, Global Q has actual shape (B, N, G, H) although we allocated space of (B, N, S, H)
  // When max_num_global == 0, these pointers are not used in GEMM so the value does not matter.
  const T* global_q = (format == 1 ? v + elements : qkv + 5 * elements);
  const T* global_k = (format == 1 ? global_q + elements : qkv + 3 * elements);
  const T* global_v = (format == 1 ? global_k + elements : qkv + 4 * elements);

  // Q*K' are scaled by 1/sqrt(H)
  const float rsqrt_head_size = 1.f / sqrt(static_cast<float>(head_size));

  T* temp_output = qkv;  // Q will be overwritten

  if (disable_compact_memory) {
    ORT_RETURN_IF_ERROR(LaunchLongformerSoftmaxSimpleKernel(
        stream,
        cublas,
        workspace,
        q,
        k,
        v,
        attention_mask,
        global_q,
        global_k,
        global_v,
        global_attention,
        global_index,
        batch_global_num,
        pinned_buffer,
        temp_output,
        rsqrt_head_size,
        batch_size,
        sequence_length,
        num_heads,
        head_size,
        window,
        element_size));
  } else {
    ORT_ENFORCE(max_num_global <= window);
    ORT_RETURN_IF_ERROR(LaunchLongformerSoftmaxKernel(
        stream,
        cublas,
        workspace,
        q,
        k,
        v,
        attention_mask,
        max_num_global,
        compact_global_q,
        global_q,
        global_k,
        global_v,
        global_attention,
        global_index,
        batch_global_num,
        pinned_buffer,
        temp_output,
        rsqrt_head_size,
        batch_size,
        sequence_length,
        num_heads,
        head_size,
        window,
        element_size));
  }

  // The temp_output is BxNxSxH, transpose it to final output BxSxNxH
  return LaunchTransCtx(stream, sequence_length, batch_size, head_size,
                        num_heads, max_threads_per_block, false, temp_output, output);
}

Status LaunchLongformerAttentionKernel(
    const cudaDeviceProp& device_prop,
    cublasHandle_t cublas,
    cudaStream_t stream,
    const void* input,
    const void* bias,
    const void* attention_mask,
    const void* global_input,
    const void* global_bias,
    const int* global_attention,
    const int* global_index,
    const int* batch_global_num,
    void* pinned_buffer,
    void* workspace,
    void* output,
    int batch_size,
    int sequence_length,
    int num_heads,
    int head_size,
    int window,
    int max_num_global,
    const size_t element_size,
    bool disable_compact_memory,
    bool use_merged_qkv_weights,
    bool use_half4) {
  size_t softmax_workspace_size = GetLongformerSoftmaxWorkspaceSize(element_size,
                                                                    batch_size,
                                                                    num_heads,
                                                                    sequence_length,
                                                                    window,
                                                                    disable_compact_memory);
  if (element_size == 2) {
    return LongformerQkvToContext(device_prop, cublas, stream,
                                  batch_size, sequence_length, num_heads, head_size, window, element_size,
                                  reinterpret_cast<const half*>(input),
                                  reinterpret_cast<const half*>(bias),
                                  reinterpret_cast<const half*>(attention_mask),
                                  reinterpret_cast<const half*>(global_input),
                                  reinterpret_cast<const half*>(global_bias),
                                  global_attention,
                                  global_index,
                                  batch_global_num,
                                  max_num_global,
                                  pinned_buffer,
                                  reinterpret_cast<half*>(workspace),
                                  reinterpret_cast<half*>(output),
                                  softmax_workspace_size,
                                  disable_compact_memory,
                                  use_merged_qkv_weights,
                                  use_half4);
  } else {
    return LongformerQkvToContext(device_prop, cublas, stream,
                                  batch_size, sequence_length, num_heads, head_size, window, element_size,
                                  reinterpret_cast<const float*>(input),
                                  reinterpret_cast<const float*>(bias),
                                  reinterpret_cast<const float*>(attention_mask),
                                  reinterpret_cast<const float*>(global_input),
                                  reinterpret_cast<const float*>(global_bias),
                                  global_attention,
                                  global_index,
                                  batch_global_num,
                                  max_num_global,
                                  pinned_buffer,
                                  reinterpret_cast<float*>(workspace),
                                  reinterpret_cast<float*>(output),
                                  softmax_workspace_size,
                                  disable_compact_memory,
                                  use_merged_qkv_weights,
                                  false);
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
