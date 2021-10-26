/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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
/**
 * Decoder transformer
 **/

#pragma once

#include "contrib_ops/cuda/fastertransformer/utils/common.h"
#include "contrib_ops/cuda/fastertransformer/utils/common_structure.h"
#include "contrib_ops/cuda/fastertransformer/utils/nccl_utils.h"
#include <cuda_runtime.h>
#include <stdlib.h>

namespace fastertransformer
{

template <typename T>
class DecodingInitParam : public AbstractParam
{
public:
  /* weights for masked_multi_head_attention */
  const T *embedding_table = nullptr;
  const T *embedding_kernel = nullptr;
  const T *embedding_bias = nullptr;

  const T *memory_tensor = nullptr;
  const int *memory_sequence_length = nullptr;

  const T *position_encoding_table = nullptr;

  LayerNormWeight<T> layernorm;

  int *output_ids = nullptr;
  int *parent_ids = nullptr;
  int *sequence_length = nullptr;
  cublasHandle_t cublas_handle;
  cublasLtHandle_t cublaslt_handle;
  cudaStream_t stream;

  // For GPT model
  int request_batch_size;
  int request_input_len;
  int request_output_len = 0;
  int max_input_len;
  const int *d_start_ids;
  const int *d_start_lengths;
  const T *d_attn_mask;

  virtual ~DecodingInitParam() {}
};

struct TransformerArguments
{
  size_t batch_size_;
  size_t seq_len_;
  size_t head_num_;
  size_t size_per_head_;
  size_t hidden_units_;
};

struct DecodingArguments : public TransformerArguments
{
  int decoder_layers_;
  int vocab_size_;
  int start_id_;
  int end_id_;
  int vocab_size_padded_;
};

struct DecodingSamplingArguments : public DecodingArguments
{
  int candidate_num_;
  float probability_threshold_;
  size_t cub_temp_storage_size_{0};
};

struct DecodingBeamsearchArguments : public DecodingArguments
{
  int beam_width_;
  int temp_storage_size_;
  float beam_search_diversity_rate_;
};

struct GptArguments : public DecodingSamplingArguments
{
  int start_len_;
  float temperature_{2.0};
  float len_penalty{1.0};
  float repetition_penalty_{1.0};
  int *vocab_mask{nullptr};
  int min_gpu_num_{1};
};

} // namespace fastertransformer