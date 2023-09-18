// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/bert/attention_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void ConcatTensorToTensor(const int tensor_add_sequence_length,
                                     const T* tensor_in,
                                     const T* tensor_add,
                                     T* tensor_out) {
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;
  const int chunk_id = blockIdx.z;

  const int all_sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int num_heads = blockDim.y;
  const int H = blockDim.x;

  // K: number of identical tensors
  // tensor_in:    K x BxNxPxH
  // tensor_add:   K x BxNxLxH
  // tensor_out:   K x BxNxTxH, where T = P + L
  const int tensor_in_sequence_length = all_sequence_length - tensor_add_sequence_length;

  const int present_SH = all_sequence_length * H;
  const int present_NSH = num_heads * present_SH;
  int out_offset = b * present_NSH + n * present_SH + s * H + h + chunk_id * (present_NSH * batch_size);
  if (s < tensor_in_sequence_length) {
    const int past_SH = tensor_in_sequence_length * H;
    const int past_NSH = num_heads * past_SH;
    const int in_offset = b * past_NSH + n * past_SH + s * H + h + chunk_id * (past_NSH * batch_size);
    tensor_out[out_offset] = tensor_in[in_offset];
  } else if (s < all_sequence_length) {
    const int SH = tensor_add_sequence_length * H;
    const int NSH = num_heads * SH;
    const int in_offset = b * NSH + n * SH + (s - tensor_in_sequence_length) * H + h + chunk_id * (NSH * batch_size);
    tensor_out[out_offset] = tensor_add[in_offset];
  }
}

template <typename T>
__global__ void ConcatTensorToTensorLarge(const int tensor_add_sequence_length,
                                          const int H,
                                          const T* tensor_in,
                                          const T* tensor_add,
                                          T* tensor_out) {
  // Use when (H*)*num_heads > 1024
  int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;
  const int chunk_id = blockIdx.z;

  const int all_sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int num_heads = blockDim.y;
  const int stride = blockDim.x;

  // K: number of identical tensor
  // tensor_in:    K x BxNxPxH
  // tensor_add:   K x BxNxLxH
  // tensor_out:   K x BxNxTxH
  const int tensor_in_sequence_length = all_sequence_length - tensor_add_sequence_length;

  const int present_SH = all_sequence_length * H;
  const int present_NSH = num_heads * present_SH;
  while (h < H) {
    int out_offset = b * present_NSH + n * present_SH + s * H + h + chunk_id * (present_NSH * batch_size);
    if (s < tensor_in_sequence_length) {
      const int past_SH = tensor_in_sequence_length * H;
      const int past_NSH = num_heads * past_SH;
      const int in_offset = b * past_NSH + n * past_SH + s * H + h + chunk_id * (past_NSH * batch_size);
      tensor_out[out_offset] = tensor_in[in_offset];
    } else if (s < all_sequence_length) {
      const int SH = tensor_add_sequence_length * H;
      const int NSH = num_heads * SH;
      const int in_offset = b * NSH + n * SH + (s - tensor_in_sequence_length) * H + h + chunk_id * (NSH * batch_size);
      tensor_out[out_offset] = tensor_add[in_offset];
    }

    h += stride;
  }
}

Status LaunchConcatTensorToTensor(cudaStream_t stream,
                                  const int all_sequence_length,
                                  const int sequence_length,
                                  const int batch_size,
                                  const int head_size,
                                  const int num_heads,
                                  const int max_threads_per_block,
                                  const int matrix_num,
                                  const float* tensor_in,
                                  const float* tensor_add,
                                  float* tensor_out) {
  const dim3 grid(all_sequence_length, batch_size, matrix_num);
  if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      ConcatTensorToTensor<float2><<<grid, block, 0, stream>>>(sequence_length,
                                                               reinterpret_cast<const float2*>(tensor_in),
                                                               reinterpret_cast<const float2*>(tensor_add),
                                                               reinterpret_cast<float2*>(tensor_out));
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<float2><<<grid, block, 0, stream>>>(sequence_length,
                                                                    H,
                                                                    reinterpret_cast<const float2*>(tensor_in),
                                                                    reinterpret_cast<const float2*>(tensor_add),
                                                                    reinterpret_cast<float2*>(tensor_out));
    }
  } else {
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      ConcatTensorToTensor<float><<<grid, block, 0, stream>>>(sequence_length, tensor_in, tensor_add, tensor_out);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<float><<<grid, block, 0, stream>>>(sequence_length,
                                                                   head_size,
                                                                   tensor_in,
                                                                   tensor_add,
                                                                   tensor_out);
    }
  }
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchConcatTensorToTensor(cudaStream_t stream,
                                  const int all_sequence_length,
                                  const int sequence_length,
                                  const int batch_size,
                                  const int head_size,
                                  const int num_heads,
                                  const int max_threads_per_block,
                                  const int matrix_num,
                                  const half* tensor_in,
                                  const half* tensor_add,
                                  half* tensor_out) {
  const dim3 grid(all_sequence_length, batch_size, matrix_num);
  if (0 == (head_size % 4)) {
    const int H = head_size / 4;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      ConcatTensorToTensor<float2><<<grid, block, 0, stream>>>(sequence_length,
                                                               reinterpret_cast<const float2*>(tensor_in),
                                                               reinterpret_cast<const float2*>(tensor_add),
                                                               reinterpret_cast<float2*>(tensor_out));
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<float2><<<grid, block, 0, stream>>>(sequence_length,
                                                                    H,
                                                                    reinterpret_cast<const float2*>(tensor_in),
                                                                    reinterpret_cast<const float2*>(tensor_add),
                                                                    reinterpret_cast<float2*>(tensor_out));
    }
  } else if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      ConcatTensorToTensor<half2><<<grid, block, 0, stream>>>(sequence_length,
                                                              reinterpret_cast<const half2*>(tensor_in),
                                                              reinterpret_cast<const half2*>(tensor_add),
                                                              reinterpret_cast<half2*>(tensor_out));
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<half2><<<grid, block, 0, stream>>>(sequence_length,
                                                                   H,
                                                                   reinterpret_cast<const half2*>(tensor_in),
                                                                   reinterpret_cast<const half2*>(tensor_add),
                                                                   reinterpret_cast<half2*>(tensor_out));
    }
  } else {  // this should be an "odd" case. probably not worth catching it in the half2 kernel.
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      ConcatTensorToTensor<half><<<grid, block, 0, stream>>>(sequence_length, tensor_in, tensor_add, tensor_out);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<half><<<grid, block, 0, stream>>>(sequence_length,
                                                                  head_size,
                                                                  tensor_in,
                                                                  tensor_add,
                                                                  tensor_out);
    }
  }
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchConcatPastToPresent(cudaStream_t stream,
                                 const int all_sequence_length,
                                 const int sequence_length,
                                 const int batch_size,
                                 const int head_size,
                                 const int num_heads,
                                 const int max_threads_per_block,
                                 const float* past,
                                 const float* k_v,
                                 float* present) {
  return LaunchConcatTensorToTensor(
      stream,
      all_sequence_length,
      sequence_length,
      batch_size,
      head_size,
      num_heads,
      max_threads_per_block,
      2,
      past,
      k_v,
      present);
}

Status LaunchConcatPastToPresent(cudaStream_t stream,
                                 const int all_sequence_length,
                                 const int sequence_length,
                                 const int batch_size,
                                 const int head_size,
                                 const int num_heads,
                                 const int max_threads_per_block,
                                 const half* past,
                                 const half* k_v,
                                 half* present) {
  return LaunchConcatTensorToTensor(
      stream,
      all_sequence_length,
      sequence_length,
      batch_size,
      head_size,
      num_heads,
      max_threads_per_block,
      2,
      past,
      k_v,
      present);
}

#ifndef USE_ROCM // exclude from hipify
template <typename T>
Status ConcatPastToPresent(int batch_size, int num_heads, int qk_head_size, int v_head_size,
                           int sequence_length, int total_sequence_length, bool pass_past_in_kv,
                           cudaStream_t stream,
                           int max_threads_per_block,
                           AttentionData<T>& data,
                           QkvData<T>& qkv) {
  // Concat past key value to present (2xBxNxLxH), where L is kv_sequence_length and T is total_sequence_length.
  // past_k (BxNxPxH) + k (BxNxLxH) => present_k (BxNxTxH)
  // past_v (BxNxPxH) + v (BxNxLxH) => present_v (BxNxTxH)
  // When there is past state, the head size for Q/K/V shall be same: H == H_v.

  if (nullptr != data.present) {
    assert(qkv.format == AttentionQkvFormat::Q_K_V_BNSH || qkv.format == AttentionQkvFormat::Q_K_V_BNSH_QKV_BS3NH);
    ORT_RETURN_IF_ERROR(
        LaunchConcatPastToPresent(
            stream, total_sequence_length, sequence_length, batch_size, qk_head_size, num_heads,
            max_threads_per_block, data.past, qkv.k, data.present));

    // Update pointers to present_k and present_v.
    qkv.k = data.present;
    qkv.v = data.present + batch_size * num_heads * total_sequence_length * qk_head_size;
  }

  if (nullptr != data.past_key || nullptr != data.present_key) {
    if (nullptr != data.past_key && nullptr == data.present_key) {
      qkv.k = const_cast<T*>(data.past_key);
      qkv.v = const_cast<T*>(data.past_value);
    } else if (nullptr == data.past_key && nullptr != data.present_key) {
      if (qkv.format == AttentionQkvFormat::Q_K_V_BNSH) {
        qkv.k = data.present_key;
        qkv.v = data.present_value;
      } else {
        assert(qkv.format == AttentionQkvFormat::Q_K_V_BSNH);
        qkv.k = data.temp_k_workspace;
        qkv.v = data.temp_v_workspace;
      }
    } else if (pass_past_in_kv) {
      // past_key and past_value are used directly as key and value in attention computations
      qkv.k = const_cast<T*>(data.past_key);
      qkv.v = const_cast<T*>(data.past_value);

      // This path has a memory copy from past_key and past_value to present_key and present_value
      // Avoid this path since the memory copy is unnecessary because past_key == present_key and
      // past_value == present_value
      int64_t k_size = (int64_t)batch_size * num_heads * total_sequence_length * qk_head_size;
      int64_t v_size = (int64_t)batch_size * num_heads * total_sequence_length * v_head_size;
      cudaMemcpyAsync(data.present_key, data.past_key, k_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(data.present_value, data.past_value, v_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    } else {
      ORT_RETURN_IF_ERROR(
          LaunchConcatTensorToTensor(stream, total_sequence_length, sequence_length,
                                     batch_size, qk_head_size, num_heads,
                                     max_threads_per_block, 1, data.past_key, qkv.k, data.present_key));
      ORT_RETURN_IF_ERROR(
          LaunchConcatTensorToTensor(stream, total_sequence_length, sequence_length,
                                     batch_size, v_head_size, num_heads,
                                     max_threads_per_block, 1, data.past_value, qkv.v, data.present_value));
      // Update pointers to present_k and present_v.
      qkv.k = data.present_key;
      qkv.v = data.present_value;
    }
  }

  return CUDA_CALL(cudaGetLastError());
}

// Template Instantiation
template Status ConcatPastToPresent<float>(int batch_size, int num_heads, int qk_head_size, int v_head_size,
                                           int sequence_length, int total_sequence_length, bool pass_past_in_kv,
                                           cudaStream_t stream,
                                           int max_threads_per_block,
                                           AttentionData<float>& data,
                                           QkvData<float>& qkv);

template Status ConcatPastToPresent<half>(int batch_size, int num_heads, int qk_head_size, int v_head_size,
                                          int sequence_length, int total_sequence_length, bool pass_past_in_kv,
                                          cudaStream_t stream,
                                          int max_threads_per_block,
                                          AttentionData<half>& data,
                                          QkvData<half>& qkv);
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
