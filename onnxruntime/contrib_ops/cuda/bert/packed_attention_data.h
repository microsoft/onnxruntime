// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cuda/bert/packed_attention_workspace.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
struct PackedAttentionData {
  T* gemm_buffer;
  const T* bias;
  const T* attention_bias;
  const int32_t* token_offset;
  const int32_t* cumulative_sequence_length;

  T* workspace;
  T* output;

  void* fused_runner;

  bool use_memory_efficient_attention;
  PackedAttentionWorkspaceRecipe workspace_recipe;
};

template <typename T>
struct PackedMultiHeadAttentionData {
  const T* query;
  const T* key;
  const T* value;
  const T* bias;
  const T* attention_bias;

  const int32_t* token_offset;
  const int32_t* cumulative_sequence_length;

  AttentionQkvFormat source_qkv_format;

  bool no_qkv_workspace;
  T* workspace;
  T* output;

  void* fused_runner;

  bool use_flash_attention;
  bool use_memory_efficient_attention;
  PackedAttentionWorkspaceRecipe workspace_recipe;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
