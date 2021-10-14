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
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "contrib_ops/cuda/fastertransformer/utils/arguments.h"
#include <assert.h>

namespace fastertransformer
{

template <typename T>
void add_QKV_bias_transpose_kernelLauncher(
  T* q_buf,
  T* k_buf,
  T* v_buf,
  T* Q,
  const T* bias_Q,
  T* K,
  const T* bias_K,
  T* V,
  const T* bias_V,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const int size_per_head,
  cudaStream_t stream);

template <typename T>
void add_fusedQKV_bias_transpose_kernelLauncher(
  T* q_buf,
  T* k_buf,
  T* v_buf,
  T* QKV,
  const T* qkv_bias,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const int size_per_head,
  cudaStream_t stream);
  
template <typename T>
void attn_softmax_kernelLauncher(
  T* buffer,
  const T* attr_mask,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const T scalar,
  cudaStream_t stream);

template <typename T>
void transpose_kernelLauncher(
  T* dst,
  T* src,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const int size_per_head,
  cudaStream_t stream);

} // namespace fastertransformer