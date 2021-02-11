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
// (3) Maximum number of global tokens <= one-sided attention window

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

#define CHECK(expr)         \
  if (!CUBLAS_CALL(expr)) { \
    return false;           \
  }

#define CHECK_CUDA(expr)  \
  if (!CUDA_CALL(expr)) { \
    return false;         \
  }

constexpr int MAX_LONGFORMER_BATCH_SIZE = 128;

namespace onnxruntime {
namespace contrib {
namespace cuda {

size_t GetScratch1Size(size_t element_size, int batch_size, int num_heads, int sequence_length, int window) {
  return (5 * sequence_length - 3 * window) * window * num_heads * batch_size * element_size;
}

// Denote: batch size (B), sequence length (S), number of heads (N), dimension per head (H), max number of global tokens (G)
//
// Workspace layout (by default, the data type T is float or half):
//   [SoftmaxSpace: see below] [Q:BxNxSxH] [K:BxNxSxH] [V:BxNxSxH] [Global_Q:BxNxSxH] [Global_K:BxNxSxH] [Global_V:BxNxSxH]
// where Global_Q, Global_K and Global_V are optional. They are not allocated when there is no any global token.
//
// It is feasible to use compact format for Global_Q with shape BxNxGxH to save space. We do not use compact format for now.
//
// SoftmaxSpace layout:
//   [Global_Idx: int BxS][batch_global_num: int BxS][sequence_index: int BxS][tmp_storage: int 1024x1]
//                                                                            [scratch1: (5S-3W)*W*N*B][scratch2: size_t 20]
//
// Temporary storage for global index calculation could use up to total space of scratch1, scratch2, Q and K. It will
// be overwritten by data of scratch1. Note that tmp_storage and scratch1 have same start address.
//
// Scratch1 has 5 buffers for local and global attention calculation.
// Scratch2 has 5 input pointers, 5 output pointers, 5 buffer sizes and 5 strides related to scratch1.
// 
// Allocated size could be slightly larger than needed: batch_global_num uses only Bx1 and allocated BxS.

size_t GetLongformerSoftmaxWorkspaceSize(
    size_t element_size,
    int batch_size,
    int num_heads,
    int sequence_length,
    int window) {
  size_t temp_size = sizeof(int) * 1024;
  size_t scratch1_size = GetScratch1Size(element_size, batch_size, num_heads, sequence_length, window);
  size_t scratch2_size = 10 * (sizeof(void*) + sizeof(size_t));
  return 3 * batch_size * sequence_length * sizeof(int) + std::max(scratch1_size + scratch2_size, temp_size);
}

size_t GetLongformerAttentionWorkspaceSize(
    size_t element_size,
    int batch_size,
    int num_heads,
    int head_size,
    int sequence_length,
    int max_num_global,
    int window) {
  size_t softmax_size = GetLongformerSoftmaxWorkspaceSize(element_size, batch_size, num_heads, sequence_length, window);
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
size_t BuildGlobalIndex(cudaStream_t stream, const int* global_attention, int batch_size, int sequence_length, void* workspace) {
  int* global_index = reinterpret_cast<int*>(workspace);
  int* batch_global_num = global_index + batch_size * sequence_length;  // Number of global tokens in each batch, shape is (batch_size)
  int* sequence_index = batch_global_num + batch_size * sequence_length;
  int* tmp_storage = sequence_index + batch_size * sequence_length;

  InitSequenceIndexKernel<<<batch_size, 128, 0, stream>>>(sequence_index, sequence_length);

  // Determine temporary device storage requirements
  // Find the global attention indices and number of global attention tokens
  size_t temp_storage_bytes = 0;
  cub::DevicePartition::Flagged(NULL, temp_storage_bytes, sequence_index,
                                global_attention, global_index, batch_global_num, sequence_length, stream);

  for (int i = 0; i < batch_size; ++i) {
    cub::DevicePartition::Flagged(reinterpret_cast<void*>(tmp_storage), temp_storage_bytes, sequence_index,
                                  global_attention + i * sequence_length, global_index + i * sequence_length,
                                  batch_global_num + i, sequence_length, stream);
  }

  return temp_storage_bytes;
}

template <typename T, int blockSize>
__launch_bounds__(blockSize)
    __global__ void LongformerSoftmaxKernel(const int* global_attention,
                                            const int* global_index,
                                            const int* batch_global_num,
                                            void* input_pointers,
                                            const T* attention_mask,
                                            float scaler,
                                            int dim0,
                                            int sequence_length,
                                            int window,
                                            int num_heads) {
  typedef cub::BlockReduce<float, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage block_reduce_temp;
  __shared__ float max_shared;
  __shared__ float sum_shared;

  int tid = threadIdx.x;
  const int batch_index = blockIdx.x / dim0;
  const int row_index = blockIdx.x % sequence_length;
  const int head_index = (blockIdx.x / sequence_length) % num_heads;

  // Adjust the pointers for the batch
  const T* mask_block = attention_mask + sequence_length * batch_index;
  const int* global_index_block = global_index + sequence_length * batch_index;
  const int global_num = batch_global_num[batch_index];

  size_t* p_inputs = (size_t*)(input_pointers);
  size_t* p_outputs = (size_t*)(input_pointers) + 5;
  size_t* input_sizes = (size_t*)(input_pointers) + 10;
  size_t* input_strides = (size_t*)(input_pointers) + 15;

  const T* inputs[5];
  T* outputs[5];
  for (int i = 0; i < 5; ++i) {
    inputs[i] = (T*)p_inputs[i] + batch_index * num_heads * input_sizes[i];
    outputs[i] = (T*)p_outputs[i] + batch_index * num_heads * input_sizes[i];
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

  // Calculate max input
  float max_input = -CUDART_INF_F;

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
    if (local_global > global_num) local_global = global_num;
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

    for (int i = local_start + tid, ii = col_start + tid; i < local_end; i += blockSize, ii += blockSize) {
      float x = input_block[i];
      x = x * scaler + (float)mask_block[ii];
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

    // __syncthreads();
    float max_blk = BlockReduce(block_reduce_temp).Reduce(max_input, cub::Max());
    if (tid == 0) {
      max_shared = max_blk;
    }
    __syncthreads();

    for (int i = local_start + tid, ii = col_start + tid; i < local_end; i += blockSize, ii += blockSize) {
      float x = input_block[i];
      x = expf((x)*scaler + (float)mask_block[ii] - max_shared);
      sum_input += x;
    }

    if (input_global != nullptr) {
      for (int i = tid, ii = col_start + tid; i < local_global; i += blockSize, ii += blockSize) {
        float x = input_global[global_index_block[i]];
        x = expf((x)*scaler + (float)mask_block[ii] - max_shared);
        sum_input += x;
      }
    }

    // __syncthreads();
    float sum_blk = BlockReduce(block_reduce_temp).Reduce(sum_input, cub::Sum());
    if (tid == 0) {
      sum_shared = sum_blk;
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

    for (int i = local_start + tid, ii = col_start + tid; i < local_end; i += blockSize, ii += blockSize) {
      float x = input_block[i];
      x = expf((x)*scaler + (float)mask_block[ii] - max_shared);
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

    // __syncthreads();
    float max_blk = BlockReduce(block_reduce_temp).Reduce(max_input, cub::Max());
    if (tid == 0) {
      max_shared = max_blk;
    }
    __syncthreads();

    for (int i = tid; i < sequence_length; i += blockSize) {
      float x = input_block[i];
      x = expf((x)*scaler + (float)mask_block[i] - max_shared);
      sum_input += x;
    }

    // __syncthreads();
    float sum_blk = BlockReduce(block_reduce_temp).Reduce(sum_input, cub::Sum());
    if (tid == 0) {
      sum_shared = sum_blk;
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

bool launchSoftmaxKernel(
    cudaStream_t stream,
    cublasHandle_t cublas,
    void* workspace,
    size_t softmax_workspace_size,
    const void* q,                // transposed Q with shape (B, N, S, H)
    const void* k,                // transposed K with shape (B, N, S, H)
    const void* v,                // transposed V with shape (B, N, S, H)
    const void* attention_mask,   // attention mask with shape (B, S), with value 0 not masked and -10000 masked.
    const void* global_q,         // Q for global tokens with shape (B, N, S, H).
    const void* global_k,         // K for global tokens with shape (B, N, S, H)
    const void* global_v,         // V for global tokens with shape (B, N, S, H)
    const int* global_attention,  // global attention with shape (B, S), with value 0 for local attention and 1 for global attention.
    void* output,                 // output with shape (B, N, S, H)
    float scaler,                 // scalar
    int batch_size,               // batch size
    int sequence_length,          // sequence length
    int num_heads,                // number of heads
    int head_size,                // hidden size per head
    int window,                   // one sided window size
    int max_num_global,           // maximum number of global tokens (G) in all batches
    size_t element_size) {        // size of element: 2 for half, and 4 for float
  if (batch_size > MAX_LONGFORMER_BATCH_SIZE) {
    ORT_THROW("LongformerAttention CUDA operator does not support batch size > 128.");
  }

  if (max_num_global > window) {
    ORT_THROW("LongformerAttention CUDA operator does not support number of global tokens > attention window.");
  }

  bool is_fp16 = (element_size == 2);
  void* scratch1 = reinterpret_cast<char*>(workspace) + 3 * sizeof(int) * batch_size * sequence_length;

  // Build index for global tokens
  size_t temp_storage_bytes = BuildGlobalIndex(stream, global_attention, batch_size, sequence_length, workspace);
  assert(temp_storage_bytes <= softmax_workspace_size - static_cast<size_t>(3 * batch_size * sequence_length));

  int* global_index = (int*)workspace;
  int* batch_global_num = global_index + batch_size * sequence_length;

  cudaEvent_t canFreeVector;
  CHECK_CUDA(cudaEventCreate(&canFreeVector));

  int num_global[MAX_LONGFORMER_BATCH_SIZE] = {-1};
  CHECK_CUDA(cudaMemcpyAsync(&num_global[0], batch_global_num, batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream));
  for (int i = 0; i < batch_size; ++i) {
    if (num_global[i] > window) {
      ORT_THROW("LongformerAttention CUDA operator does not support number of global tokens > attention window.");
    }
  }

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
  int elements_per_batch = num_heads * sequence_length * head_size;
  int stride_per_head = sequence_length * head_size;  // stride for Q, K, V and output

  // Local attention part
  // S x S is calculated using sliding block WxW (W is one sided window size) like the following:
  //   [W][W]
  //   [W][W][W]
  //      [W][W][W]
  //         [W][W]
  // The first and last rows have 2 blocks per row, and the remaining has 3 blocks per row.
  // The calculation are splited into 3 parts. Firstly, fill the middle rows,  then the first row and finally the last row.
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
      static_cast<size_t>(w * (sequence_length - w)),  // local attends to global: global tokens are assumed to be smaller than window size
      static_cast<size_t>(w * sequence_length)};       // global attends to everything.

  size_t buffer_strides[5] = {
      static_cast<size_t>(w * 2),
      static_cast<size_t>(w * 3),
      static_cast<size_t>(w * 2),
      static_cast<size_t>(w),  // global tokens are assumed to be smaller than window size
      static_cast<size_t>(sequence_length)};

  void* buffer_pointers[11];  //0~4: input pointers; 5~9: output pointers; 10: pointer to right after those 5 buffers

  char* buffer_pointer = (char*)scratch1;
  for (int i = 0; i < 5; ++i) {
    buffer_pointers[i] = (void*)buffer_pointer;
    buffer_pointers[i + 5] = (void*)buffer_pointer;  // output pointer is same as input
    buffer_pointer += buffer_sizes[i] * num_heads * batch_size * element_size;
  }
  assert(buffer_pointer == (char*)scratch1 + GetScratch1Size(element_size, batch_size, num_heads, sequence_length, window));

  buffer_pointers[10] = buffer_pointer;

  // Copy to a continues buffer first so that we only need call cudaMemcpyAsync once
  constexpr size_t totalBytes = 10 * (sizeof(size_t) + sizeof(void*));
  char temp_buffer[totalBytes];
  memcpy(temp_buffer, &buffer_pointers[0], 10 * sizeof(void*));
  memcpy(temp_buffer + 10 * sizeof(void*), &buffer_sizes[0], 5 * sizeof(size_t));
  memcpy(temp_buffer + 10 * sizeof(void*) + 5 * sizeof(size_t), &buffer_strides[0], 5 * sizeof(size_t));
  CHECK_CUDA(cudaMemcpyAsync(buffer_pointers[10], temp_buffer, totalBytes, cudaMemcpyHostToDevice, stream));

  // Create an event to make sure the async copy is finished before this function exits, otherwise temp_buffer will disappear.
  CHECK_CUDA(cudaEventRecord(canFreeVector, stream));

  // Local attention part
  {
    if (middle_count > 0) {
      for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_heads; ++j) {
          void* q_head = (char*)q + (i * elements_per_batch + j * sequence_length * head_size + w * head_size) * element_size;
          void* k_head = (char*)k + (i * elements_per_batch + j * sequence_length * head_size) * element_size;
          void* qk_head = (char*)buffer_pointers[1] + (i * num_heads + j) * buffer_sizes[1] * element_size;
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

    void* q_head = (char*)q + (last_block * w * head_size) * element_size;
    void* k_head = (char*)k + ((last_block - 1) * w * head_size) * element_size;

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
    if (num_global[i] > 0) {
      void* q_batch = (char*)q + (i * elements_per_batch + w * head_size) * element_size;
      void* k_batch = (char*)k + (i * elements_per_batch) * element_size;
      void* qk_batch = (char*)buffer_pointers[3] + (i * buffer_sizes[3]) * num_heads * element_size;

      // Local tokens attending global tokens
      CHECK(cublasGemmStridedBatchedEx(cublas,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       num_global[i],        // m
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

      void* global_q_batch = (char*)global_q + (i * elements_per_batch) * element_size;  // For compact format: replace elements_per_batch by num_heads * max_num_global * head_size
      void* global_k_batch = (char*)global_k + (i * elements_per_batch) * element_size;
      qk_batch = (char*)buffer_pointers[4] + (i * buffer_sizes[4] * num_heads) * element_size;

      // Global tokens attending everything
      // This GEMMs need to be last to make sure all global token entries are re-written.
      CHECK(cublasGemmStridedBatchedEx(cublas,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       sequence_length,  // m
                                       num_global[i],    // n
                                       head_size,        // k
                                       alpha,            // alpha
                                       global_k_batch,   // A
                                       Atype,            // A type
                                       head_size,        // lda
                                       stride_per_head,  // strideA
                                       global_q_batch,   // B
                                       Btype,            // B type
                                       head_size,        // ldb
                                       stride_per_head,  // strideB. For compact format: max_num_global * head_size.
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

  int dim0 = sequence_length * num_heads;
  int dim1 = sequence_length;

  const int blockSize = 64;
  const int gridSize = batch_size * num_heads * sequence_length;
  if (is_fp16) {
    LongformerSoftmaxKernel<__half, blockSize><<<gridSize, blockSize, 0, stream>>>(
        global_attention,
        global_index,
        batch_global_num,
        buffer_pointers[10],
        static_cast<const __half*>(attention_mask),
        scaler, dim0, dim1, window, num_heads);
  } else {
    LongformerSoftmaxKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(
        global_attention,
        global_index,
        batch_global_num,
        buffer_pointers[10],
        static_cast<const float*>(attention_mask),
        scaler, dim0, dim1, window, num_heads);
  }

  // Run the matrix multiply: output = softmax_out * v
  //   softmax_out: B x N x S x S
  //             v: B x N x S x H
  //      attn_out: B x N x S x H
  // Calculation uses sliding blocks in a way similar to local attention part.

  {
    if (middle_count > 0) {
      for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_heads; ++j) {
          void* v_head = (char*)v + (i * elements_per_batch + j * head_size * sequence_length) * element_size;
          void* prob_head = (char*)buffer_pointers[5 + 1] + (i * num_heads * buffer_sizes[1] + j * buffer_sizes[1]) * element_size;
          void* out_head = (char*)output + (i * elements_per_batch + j * head_size * sequence_length + w * head_size) * element_size;
          CHECK(cublasGemmStridedBatchedEx(cublas,
                                           CUBLAS_OP_N,
                                           CUBLAS_OP_N,
                                           head_size,               // m
                                           w,                       // n
                                           3 * w,                   // k
                                           alpha,                   // alpha
                                           v_head,                  // A
                                           Atype,                   // A type
                                           head_size,               // lda
                                           w * head_size,           // strideA
                                           prob_head,               // B
                                           Btype,                   // B type
                                           (int)buffer_strides[1],  // ldb
                                           3 * w * w,               // strideB
                                           beta_0,                  // beta
                                           out_head,                // C
                                           Ctype,                   // C type
                                           head_size,               // ldc
                                           w * head_size,           // strideC
                                           middle_count,            // batch count
                                           resultType,
                                           algo));
        }
      }
    }

    CHECK(cublasGemmStridedBatchedEx(cublas,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     head_size,               // m
                                     w,                       // n
                                     2 * w,                   // k
                                     alpha,                   // alpha
                                     v,                       // A
                                     Atype,                   // A type
                                     head_size,               // lda
                                     stride_per_head,         // strideA
                                     buffer_pointers[5 + 0],  // B
                                     Btype,                   // B type
                                     (int)buffer_strides[0],  // ldb
                                     buffer_sizes[0],         // strideB
                                     beta_0,                  // beta
                                     output,                  // C
                                     Ctype,                   // C type
                                     head_size,               // ldc
                                     stride_per_head,         // strideC
                                     batch_size * num_heads,  // batch count
                                     resultType,
                                     algo));

    void* v_head = (char*)v + (last_block - 1) * w * head_size * element_size;
    void* out_head = (char*)output + last_block * w * head_size * element_size;

    CHECK(cublasGemmStridedBatchedEx(cublas,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     head_size,               // m
                                     w,                       // n
                                     2 * w,                   // k
                                     alpha,                   // alpha
                                     v_head,                  // A
                                     Atype,                   // A type
                                     head_size,               // lda
                                     stride_per_head,         // strideA
                                     buffer_pointers[5 + 2],  // B
                                     Btype,                   // B type
                                     (int)buffer_strides[2],  // ldb
                                     buffer_sizes[2],         // strideB
                                     beta_0,                  // beta
                                     out_head,                // C
                                     Ctype,                   // C type
                                     head_size,               // ldc
                                     stride_per_head,         // strideC
                                     batch_size * num_heads,  // batch count
                                     resultType,
                                     algo));
  }

  for (int i = 0; i < batch_size; ++i) {
    if (num_global[i] > 0) {
      int glob_longdim_mm = sequence_length - 2 * w;

      void* v_head = (char*)v + (i * elements_per_batch) * element_size;
      void* prob_head = (char*)buffer_pointers[5 + 3] + (i * buffer_sizes[3] * num_heads + w * buffer_strides[3]) * element_size;
      void* out_head = (char*)output + (i * elements_per_batch + 2 * w * head_size) * element_size;

      CHECK(cublasGemmStridedBatchedEx(cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       head_size,               // m
                                       glob_longdim_mm,         // n
                                       num_global[i],           // k
                                       alpha,                   // alpha
                                       v_head,                  // A
                                       Atype,                   // A type
                                       head_size,               // lda
                                       stride_per_head,         // strideA
                                       prob_head,               // B
                                       Btype,                   // B type
                                       (int)buffer_strides[3],  // ldb
                                       buffer_sizes[3],         // strideB
                                       beta_1,                  // beta
                                       out_head,                // C
                                       Ctype,                   // C type
                                       head_size,               // ldc
                                       stride_per_head,         // strideC
                                       num_heads,               // batch count
                                       resultType,
                                       algo));

      // Global tokens
      v_head = (char*)global_v + (i * elements_per_batch) * element_size;
      prob_head = (char*)buffer_pointers[5 + 4] + (i * buffer_sizes[4] * num_heads) * element_size;
      out_head = (char*)output + (i * elements_per_batch) * element_size;

      CHECK(cublasGemmStridedBatchedEx(cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       head_size,               // m
                                       num_global[i],           // n
                                       sequence_length,         // k: re-write entries completely
                                       alpha,                   // alpha
                                       v_head,                  // A
                                       Atype,                   // A type
                                       head_size,               // lda
                                       stride_per_head,         //strideA
                                       prob_head,               // B
                                       Btype,                   // B type
                                       (int)buffer_strides[4],  // ldb
                                       buffer_sizes[4],         // strideB
                                       beta_0,                  // beta: overwrite
                                       out_head,                // C: assumes global tokens are at the beginning of sequence
                                       Ctype,                   // C type
                                       head_size,               // ldc
                                       stride_per_head,         // strideC
                                       num_heads,               // batch count
                                       resultType,
                                       algo));
    }
  }

  // Make sure we do not exit function before async copy is finished
  CHECK_CUDA(cudaStreamWaitEvent(stream, canFreeVector, 0));
  CHECK_CUDA(cudaEventDestroy(canFreeVector));

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
  size_t softmax_workspace_size = GetLongformerSoftmaxWorkspaceSize(element_size, batch_size, num_heads, sequence_length, window);
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
          global_q,          // Transposed global Q with shape B x N x S x H.
          global_k,          // Transposed global K with shape B x N x S x H
          global_v,          // Transposed global V with shape B x N x S x H
          global_attention,  // Global attention flags with shape B x S
          temp_output,       // Output with shape B x N x S x H
          rsqrt_head_size,   // Scaler
          batch_size,        // Batch size
          sequence_length,   // Sequence length
          num_heads,         // Number of attention heads
          head_size,         // Hidden size per head
          window,            // Half (one-sided) window size
          max_num_global,    // Maximum number of global tokens (G)
          element_size)) {
    return false;
  }

  // The temp_output is BxNxSxH, transpose it to final output BxSxNxH
  return LaunchTransCtx(stream, sequence_length, batch_size, head_size, num_heads, temp_output, output);
}

bool LaunchLongformerAttentionKernel(
    const cudaDeviceProp& prop,
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
  // use default stream
  const cudaStream_t stream = nullptr;

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
