/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

namespace fastertransformer{

template <typename T>
void fusedQKV_masked_attention_dispatch(
  const T* qkv_buf, const T* qkv_bias,
  T* key_cache, T* value_cache,
  T* context_buf, const bool* finished, int max_batch_size, int inference_batch_size, 
  int head_num, int size_per_head, const int step, const int max_seq_len, cudaStream_t stream);

template <typename T>
void fusedQKV_masked_attention_dispatch_v2(
  const T* qkv_buf, const T* qkv_bias,
  T* key_cache, T* value_cache,
  T* context_buf, const bool* finished, int max_batch_size, int inference_batch_size, 
  int head_num, int size_per_head, const int step, const int max_seq_len, 
  const int max_input_len, const int* input_lengths, cudaStream_t stream);

template <typename T>
void masked_attention_dispatch(
  T* key_buf, T* value_buf,
  T* query_buf, const T* self_Q_bias, 
  T* key_cache, const T* self_K_bias, T* value_cache, const T* self_V_bias,
  T* context_buf, const bool* finished, int max_batch_size, int inference_batch_size,
  int head_num, int size_per_head, const int step, const int max_seq_len, cudaStream_t stream);

template <typename T>
void cross_attention_dispatch(T* query_buf, const T* Q_bias, 
  T* key_cache, const T* K_bias, T* value_cache, const T* V_bias, const int* length,
  T* context_buf, const bool* finished,
  int batch_size, int head_num, int size_per_head, int step, int seq_len, cudaStream_t stream);

template <typename T>
void fusedQKV_masked_attention_kernelLauncher(
  const T* qkv_buf,
  const T* qkv_bias,
  T* k_cache,
  T* v_cache,
  T* output,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const int size_per_head,
  const int max_seq_len,
  cudaStream_t stream
);

template<typename T>
void transpose_4d_kernelLauncher(T* dst, T* src,
  const int local_batch_size,
  const int seq_len,
  const int size_per_head,
  const int local_hidden_units,
  const int local_head_num,
  const int batch_size,
  const int ite,
  cudaStream_t stream
);

template<typename T>
void transpose_4d_batch_major_kernelLauncher(T* k_dst, T* v_dst,
                                 const T* k_src, const T* v_src,
                                 const int local_batch_size,
                                 const int seq_len,
                                 const int max_seq_len,
                                 const int size_per_head,
                                 const int local_head_num,
                                 cudaStream_t stream);

}
