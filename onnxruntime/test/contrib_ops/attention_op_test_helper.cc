// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include <mutex>
#include "gtest/gtest.h"
#include "core/common/common.h"
#include "test/contrib_ops/attention_op_test_helper.h"
#include "test/util/include/tensors_from_file.h"

namespace onnxruntime {
namespace test {

static void LoadTensor(const std::string& name, std::vector<float>& values) {
  static std::unordered_map<std::string, std::vector<float>> tensors;
  static std::once_flag load_once_flag;
  std::call_once(load_once_flag, []() {
    const std::string test_data_path("testdata/attention/attention_test_data.txt");
    LoadTensorsFromFile(test_data_path, tensors);

    const std::string packed_mha_test_data_path("testdata/attention/packed_multihead_attention_test_data.txt");
    LoadTensorsFromFile(packed_mha_test_data_path, tensors);
  });

  auto it = tensors.find(name);
  if (it == tensors.end()) {
    ORT_THROW("Failed to find key:", name);
  }

  values = it->second;
}

void GetWeight_64_3_64(std::vector<float>& weight_data) {
  LoadTensor("Weight_64_3_64.weight_data", weight_data);
}

void GetBias_3_64(std::vector<float>& bias_data) {
  LoadTensor("Bias_3_64.bias_data", bias_data);
}

void SampleAttentionWeight(std::vector<float>& data, std::vector<float>& output,
                           int elements, int start_offset, int step) {
  int data_length = static_cast<int>(data.size());

  output.resize(elements);
  int index = start_offset;
  for (int i = 0; i < elements; i++) {
    index = index % data_length;
    output[i] = data[index];
    index += step;
  }
}

void GetAttentionWeight(std::vector<float>& weight_data, int elements, int start_offset, int step) {
  std::vector<float> data;
  GetWeight_64_3_64(data);
  SampleAttentionWeight(data, weight_data, elements, start_offset, step);
}

void GetAttentionBias(std::vector<float>& bias_data, int elements, int start_offset, int step) {
  std::vector<float> data;
  GetBias_3_64(data);
  SampleAttentionWeight(data, bias_data, elements, start_offset, step);
}

void GetCrossAttentionData_HeadSize40(AttentionTestData& data) {
  data.hidden_size = 80;
  data.v_hidden_size = 80;
  data.num_heads = 2;
  data.batch_size = 2;
  data.sequence_length = 3;
  data.kv_sequence_length = 5;
  data.mask_type = AttentionMaskType::MASK_NONE;
  LoadTensor("CrossAttentionData_HeadSize40.query_data", data.query_data);
  LoadTensor("CrossAttentionData_HeadSize40.key_data", data.key_data);
  LoadTensor("CrossAttentionData_HeadSize40.value_data", data.value_data);
  LoadTensor("CrossAttentionData_HeadSize40.bias_data", data.bias_data);
  LoadTensor("CrossAttentionData_HeadSize40.fp32_output_data", data.fp32_output_data);
  LoadTensor("CrossAttentionData_HeadSize40.fp16_output_data", data.fp16_output_data);
}

void GetCrossAttentionData_HeadSize40_NoBias(AttentionTestData& data) {
  GetCrossAttentionData_HeadSize40(data);
  data.bias_data.clear();
  LoadTensor("CrossAttentionData_HeadSize40_NoBias.fp32_output_data", data.fp32_output_data);
  data.fp16_output_data = data.fp32_output_data;
}

void GetCrossAttentionData_Batch2_HeadSize32_RightSidePadding(AttentionTestData& data, bool is_mask_1d) {
  data.hidden_size = 64;
  data.v_hidden_size = 64;
  data.num_heads = 2;
  data.batch_size = 2;
  data.sequence_length = 2;
  data.kv_sequence_length = 3;

  if (is_mask_1d) {
    data.mask_type = AttentionMaskType::MASK_1D_KEY_SEQ_LEN;
    data.key_padding_mask_data = {1, 2};
  } else {
    data.mask_type = AttentionMaskType::MASK_2D_KEY_PADDING;
    data.key_padding_mask_data = {1, 0, 0,
                                  1, 1, 0};
  }

  data.skip_kernel_types = {AttentionKernelType::AttentionKernel_TrtFusedCrossAttention,
                            AttentionKernelType::AttentionKernel_TrtFusedAttention,
                            AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention};

  LoadTensor("CrossAttentionData_Batch2_HeadSize32_RightSidePadding.query_data", data.query_data);
  LoadTensor("CrossAttentionData_Batch2_HeadSize32_RightSidePadding.key_data", data.key_data);
  LoadTensor("CrossAttentionData_Batch2_HeadSize32_RightSidePadding.value_data", data.value_data);
  LoadTensor("CrossAttentionData_Batch2_HeadSize32_RightSidePadding.bias_data", data.bias_data);
  LoadTensor("CrossAttentionData_Batch2_HeadSize32_RightSidePadding.fp32_output_data", data.fp32_output_data);
  data.fp16_output_data = data.fp32_output_data;
}

void GetCrossAttentionData_Batch2_HeadSize32_RightSidePadding_NoBias(AttentionTestData& data, bool is_mask_1d) {
  GetCrossAttentionData_Batch2_HeadSize32_RightSidePadding(data, is_mask_1d);
  data.bias_data.clear();

  LoadTensor("CrossAttentionData_Batch2_HeadSize32_RightSidePadding_NoBias.fp32_output_data", data.fp32_output_data);
  data.fp16_output_data = data.fp32_output_data;
}

void GetCrossAttentionData_Batch1_HeadSize32_LeftSidePadding(AttentionTestData& data) {
  data.hidden_size = 32;
  data.v_hidden_size = 32;
  data.num_heads = 1;
  data.batch_size = 2;
  data.sequence_length = 2;
  data.kv_sequence_length = 3;
  data.mask_type = AttentionMaskType::MASK_2D_KEY_PADDING;
  data.key_padding_mask_data = {0, 1, 1,   // first key sequence has one padding on the left
                                0, 0, 1};  // second key sequence has two paddings on the left

  data.skip_kernel_types = {
      AttentionKernelType::AttentionKernel_TrtFusedCrossAttention,
      AttentionKernelType::AttentionKernel_TrtFusedAttention,
      AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention};

  LoadTensor("CrossAttentionData_Batch1_HeadSize32_LeftSidePadding.query_data", data.query_data);
  LoadTensor("CrossAttentionData_Batch1_HeadSize32_LeftSidePadding.key_data", data.key_data);
  LoadTensor("CrossAttentionData_Batch1_HeadSize32_LeftSidePadding.value_data", data.value_data);
  LoadTensor("CrossAttentionData_Batch1_HeadSize32_LeftSidePadding.bias_data", data.bias_data);
  LoadTensor("CrossAttentionData_Batch1_HeadSize32_LeftSidePadding.fp32_output_data", data.fp32_output_data);
  data.fp16_output_data = data.fp32_output_data;
}

void GetCrossAttentionData_Batch1_HeadSize32_LeftSidePadding_NoBias(AttentionTestData& data) {
  GetCrossAttentionData_Batch1_HeadSize32_LeftSidePadding(data);
  data.bias_data.clear();
  LoadTensor("CrossAttentionData_Batch1_HeadSize32_LeftSidePadding_NoBias.fp32_output_data", data.fp32_output_data);
  data.fp16_output_data = data.fp32_output_data;
}

void GetCrossAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedKV(AttentionTestData& data) {
  data.hidden_size = 32;
  data.v_hidden_size = 32;
  data.num_heads = 1;
  data.batch_size = 2;
  data.sequence_length = 2;
  data.kv_sequence_length = 3;
  data.mask_type = AttentionMaskType::MASK_NONE;
  // Packed KV format is only supported by TRT fused cross attention or memory efficient attention right now.
  data.skip_kernel_types = {
      AttentionKernelType::AttentionKernel_Unfused,
      AttentionKernelType::AttentionKernel_TrtFusedAttention};
  LoadTensor("CrossAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedKV.query_data", data.query_data);
  LoadTensor("CrossAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedKV.key_data", data.key_data);
  LoadTensor("CrossAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedKV.value_data", data.value_data);
  data.bias_data = {};
  LoadTensor("CrossAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedKV.kv_data", data.kv_data);
  // Do not test fp32
  data.fp32_output_data = {};
  LoadTensor("CrossAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedKV.fp16_output_data", data.fp16_output_data);
}

void GetSelfAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedQKV(AttentionTestData& data) {
  data.hidden_size = 32;
  data.v_hidden_size = 32;
  data.num_heads = 1;
  data.batch_size = 2;
  data.sequence_length = 2;
  data.kv_sequence_length = 2;
  data.mask_type = AttentionMaskType::MASK_NONE;
  // Packed QKV format is only supported by TRT fused attention or memory efficient attention right now.
  data.skip_kernel_types = {
      AttentionKernelType::AttentionKernel_Unfused,
      AttentionKernelType::AttentionKernel_TrtFusedCrossAttention};

  LoadTensor("SelfAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedQKV.query_data", data.query_data);
  LoadTensor("SelfAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedQKV.key_data", data.key_data);
  LoadTensor("SelfAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedQKV.value_data", data.value_data);
  data.bias_data = {};
  LoadTensor("SelfAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedQKV.qkv_data", data.qkv_data);
  // Do not test fp32
  data.fp32_output_data = {};
  LoadTensor("SelfAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedQKV.fp16_output_data", data.fp16_output_data);
}

void GetCrossAttentionData_HeadSize16_8(AttentionTestData& data) {
  data.hidden_size = 48;
  data.v_hidden_size = 24;
  data.num_heads = 3;
  data.batch_size = 2;
  data.sequence_length = 1;
  data.kv_sequence_length = 3;
  data.mask_type = AttentionMaskType::MASK_NONE;
  data.skip_kernel_types = {
      AttentionKernelType::AttentionKernel_TrtFusedCrossAttention,
      AttentionKernelType::AttentionKernel_TrtFusedAttention};

  LoadTensor("CrossAttentionData_HeadSize16_8.query_data", data.query_data);
  LoadTensor("CrossAttentionData_HeadSize16_8.key_data", data.key_data);
  LoadTensor("CrossAttentionData_HeadSize16_8.value_data", data.value_data);
  LoadTensor("CrossAttentionData_HeadSize16_8.bias_data", data.bias_data);
  LoadTensor("CrossAttentionData_HeadSize16_8.fp32_output_data", data.fp32_output_data);
  data.fp16_output_data = data.fp32_output_data;
}

void GetCrossAttentionData_HeadSize16_8_NoBias(AttentionTestData& data) {
  GetCrossAttentionData_HeadSize16_8(data);
  data.bias_data.clear();
  LoadTensor("CrossAttentionData_HeadSize16_8_NoBias.fp32_output_data", data.fp32_output_data);
  data.fp16_output_data = data.fp32_output_data;
}

void GetCrossAttentionData_HeadSize16(AttentionTestData& data) {
  data.hidden_size = 32;
  data.v_hidden_size = 32;
  data.num_heads = 2;
  data.batch_size = 1;
  data.sequence_length = 2;
  data.kv_sequence_length = 3;
  data.mask_type = AttentionMaskType::MASK_NONE;

  LoadTensor("CrossAttentionData_HeadSize16.query_data", data.query_data);
  LoadTensor("CrossAttentionData_HeadSize16.key_data", data.key_data);
  LoadTensor("CrossAttentionData_HeadSize16.value_data", data.value_data);
  LoadTensor("CrossAttentionData_HeadSize16.bias_data", data.bias_data);
  LoadTensor("CrossAttentionData_HeadSize16.fp32_output_data", data.fp32_output_data);
  data.fp16_output_data = data.fp32_output_data;
}

void GetCrossAttentionData_HeadSize16_NoBias(AttentionTestData& data) {
  GetCrossAttentionData_HeadSize16(data);
  data.bias_data.clear();
  LoadTensor("CrossAttentionData_HeadSize16_NoBias.fp32_output_data", data.fp32_output_data);
  data.fp16_output_data = data.fp32_output_data;
}

void GetCrossAttentionDataWithPast(AttentionTestData& data) {
  data.hidden_size = 8;
  data.v_hidden_size = 8;
  data.num_heads = 2;
  data.batch_size = 1;
  data.sequence_length = 2;
  data.kv_sequence_length = 3;
  data.mask_type = AttentionMaskType::MASK_2D_KEY_PADDING;
  data.key_padding_mask_data = {1, 1, 1};

  data.skip_kernel_types = {
      AttentionKernelType::AttentionKernel_TrtFlashAttention,
      AttentionKernelType::AttentionKernel_TrtFusedCrossAttention,
      AttentionKernelType::AttentionKernel_TrtFusedAttention,
      AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention};

  LoadTensor("CrossAttentionDataWithPast.query_data", data.query_data);
  // The past key and value data will be passed to the kernel as input 'key' and 'value'.
  LoadTensor("CrossAttentionDataWithPast.past_key_data", data.past_key_data);
  LoadTensor("CrossAttentionDataWithPast.past_value_data", data.past_value_data);
  LoadTensor("CrossAttentionDataWithPast.fp32_output_data", data.fp32_output_data);
  data.fp16_output_data = data.fp32_output_data;
}

void GetSelfAttentionData_WithPast_WithRelPosBias_ForT5(AttentionTestData& data) {
  data.hidden_size = 8;
  data.v_hidden_size = 8;
  data.num_heads = 2;
  data.batch_size = 1;
  data.sequence_length = 2;
  data.kv_sequence_length = 3;
  data.mask_type = AttentionMaskType::MASK_NONE;

  data.skip_kernel_types = {
      AttentionKernelType::AttentionKernel_TrtFlashAttention,
      AttentionKernelType::AttentionKernel_TrtFusedCrossAttention,
      AttentionKernelType::AttentionKernel_TrtFusedAttention,
      AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention,
  };

  LoadTensor("SelfAttentionData_WithPast_WithRelPosBias_ForT5.query_data", data.query_data);
  LoadTensor("SelfAttentionData_WithPast_WithRelPosBias_ForT5.key_data", data.key_data);
  LoadTensor("SelfAttentionData_WithPast_WithRelPosBias_ForT5.value_data", data.value_data);
  LoadTensor("SelfAttentionData_WithPast_WithRelPosBias_ForT5.rel_pos_bias_data", data.rel_pos_bias_data);
  data.broadcast_rel_pos_bias = false;
  LoadTensor("SelfAttentionData_WithPast_WithRelPosBias_ForT5.past_key_data", data.past_key_data);
  LoadTensor("SelfAttentionData_WithPast_WithRelPosBias_ForT5.past_value_data", data.past_value_data);
  LoadTensor("SelfAttentionData_WithPast_WithRelPosBias_ForT5.fp32_output_data", data.fp32_output_data);
  data.fp16_output_data = data.fp32_output_data;
  LoadTensor("SelfAttentionData_WithPast_WithRelPosBias_ForT5.present_key_data", data.present_key_data);
  LoadTensor("SelfAttentionData_WithPast_WithRelPosBias_ForT5.present_value_data", data.present_value_data);
  data.is_static_kv = false;
}

void GetAttentionDataCutlassRelPosBias(AttentionTestData& data) {
  data.hidden_size = 8;
  data.v_hidden_size = 8;
  data.num_heads = 2;
  data.batch_size = 1;
  data.sequence_length = 8;
  data.kv_sequence_length = 0;
  data.mask_type = AttentionMaskType::MASK_1D_KEY_SEQ_LEN_START;

  data.key_padding_mask_data = {8, 0, 8, 0, 8};

  data.skip_kernel_types = {
      AttentionKernelType::AttentionKernel_TrtFlashAttention,
      AttentionKernelType::AttentionKernel_TrtFusedCrossAttention,
      AttentionKernelType::AttentionKernel_TrtFusedAttention};

  LoadTensor("AttentionDataCutlassRelPosBias.query_data", data.query_data);
  LoadTensor("AttentionDataCutlassRelPosBias.key_data", data.key_data);
  LoadTensor("AttentionDataCutlassRelPosBias.value_data", data.value_data);
  LoadTensor("AttentionDataCutlassRelPosBias.bias_data", data.bias_data);
  LoadTensor("AttentionDataCutlassRelPosBias.rel_pos_bias_data", data.rel_pos_bias_data);
  data.broadcast_rel_pos_bias = false;
  LoadTensor("AttentionDataCutlassRelPosBias.fp16_output_data", data.fp16_output_data);
  data.fp32_output_data = {};
  data.is_static_kv = false;
}

void GetCrossAttentionData_DiffSequenceLengths(AttentionTestData& data) {
  data.hidden_size = 8;
  data.v_hidden_size = 8;
  data.num_heads = 2;
  data.batch_size = 2;
  data.sequence_length = 2;
  data.kv_sequence_length = 4;
  data.mask_type = AttentionMaskType::MASK_NONE;

  data.skip_kernel_types = {
      AttentionKernelType::AttentionKernel_TrtFlashAttention,
      AttentionKernelType::AttentionKernel_TrtFusedCrossAttention,
      AttentionKernelType::AttentionKernel_TrtFusedAttention,
      AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention,
  };

  LoadTensor("CrossAttentionData_DiffSequenceLengths.query_data", data.query_data);
  LoadTensor("CrossAttentionData_DiffSequenceLengths.key_data", data.key_data);
  LoadTensor("CrossAttentionData_DiffSequenceLengths.value_data", data.value_data);
  LoadTensor("CrossAttentionData_DiffSequenceLengths.bias_data", data.bias_data);
  LoadTensor("CrossAttentionData_DiffSequenceLengths.fp32_output_data", data.fp32_output_data);
  LoadTensor("CrossAttentionData_DiffSequenceLengths.present_key_data", data.present_key_data);
  LoadTensor("CrossAttentionData_DiffSequenceLengths.present_value_data", data.present_value_data);
  data.is_static_kv = true;
}

void GetCrossAttentionData_DiffSequenceLengths_HeadSize8(AttentionTestData& data) {
  data.hidden_size = 16;
  data.v_hidden_size = 16;
  data.num_heads = 2;
  data.batch_size = 1;
  data.sequence_length = 2;
  data.kv_sequence_length = 4;
  data.mask_type = AttentionMaskType::MASK_NONE;

  data.skip_kernel_types = {
      AttentionKernelType::AttentionKernel_TrtFlashAttention,
      AttentionKernelType::AttentionKernel_TrtFusedCrossAttention,
      AttentionKernelType::AttentionKernel_TrtFusedAttention,
      AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention,
  };

  LoadTensor("CrossAttentionData_DiffSequenceLengths_HeadSize8.query_data", data.query_data);
  LoadTensor("CrossAttentionData_DiffSequenceLengths_HeadSize8.key_data", data.key_data);
  LoadTensor("CrossAttentionData_DiffSequenceLengths_HeadSize8.value_data", data.value_data);
  LoadTensor("CrossAttentionData_DiffSequenceLengths_HeadSize8.bias_data", data.bias_data);
  LoadTensor("CrossAttentionData_DiffSequenceLengths_HeadSize8.fp32_output_data", data.fp32_output_data);
  data.fp16_output_data = data.fp32_output_data;
  LoadTensor("CrossAttentionData_DiffSequenceLengths_HeadSize8.present_key_data", data.present_key_data);
  LoadTensor("CrossAttentionData_DiffSequenceLengths_HeadSize8.present_value_data", data.present_value_data);
  data.is_static_kv = true;
}

void GetCrossAttentionData_DiffSequenceLengths_HeadSize8_NoBias(AttentionTestData& data) {
  GetCrossAttentionData_DiffSequenceLengths_HeadSize8(data);
  data.bias_data.clear();
  LoadTensor("CrossAttentionData_DiffSequenceLengths_HeadSize8_NoBias.fp32_output_data", data.fp32_output_data);
  data.fp16_output_data = data.fp32_output_data;
  LoadTensor("CrossAttentionData_DiffSequenceLengths_HeadSize8_NoBias.present_key_data", data.present_key_data);
  LoadTensor("CrossAttentionData_DiffSequenceLengths_HeadSize8_NoBias.present_value_data", data.present_value_data);
  data.is_static_kv = true;
}

void GetSelfAttentionData_WithPastAndPresent_NoMask_NoRelPosBias(AttentionTestData& data) {
  data.hidden_size = 8;
  data.v_hidden_size = 8;
  data.num_heads = 2;
  data.batch_size = 2;
  data.sequence_length = 1;
  data.kv_sequence_length = 1;
  data.mask_type = AttentionMaskType::MASK_NONE;

  data.skip_kernel_types = {
      AttentionKernelType::AttentionKernel_TrtFlashAttention,
      AttentionKernelType::AttentionKernel_TrtFusedCrossAttention,
      AttentionKernelType::AttentionKernel_TrtFusedAttention,
      AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention,
  };

  LoadTensor("SelfAttentionData_WithPastAndPresent_NoMask_NoRelPosBias.query_data", data.query_data);
  LoadTensor("SelfAttentionData_WithPastAndPresent_NoMask_NoRelPosBias.key_data", data.key_data);
  LoadTensor("SelfAttentionData_WithPastAndPresent_NoMask_NoRelPosBias.value_data", data.value_data);
  LoadTensor("SelfAttentionData_WithPastAndPresent_NoMask_NoRelPosBias.bias_data", data.bias_data);
  LoadTensor("SelfAttentionData_WithPastAndPresent_NoMask_NoRelPosBias.past_key_data", data.past_key_data);
  LoadTensor("SelfAttentionData_WithPastAndPresent_NoMask_NoRelPosBias.past_value_data", data.past_value_data);
  LoadTensor("SelfAttentionData_WithPastAndPresent_NoMask_NoRelPosBias.fp32_output_data", data.fp32_output_data);
  LoadTensor("SelfAttentionData_WithPastAndPresent_NoMask_NoRelPosBias.present_key_data", data.present_key_data);
  LoadTensor("SelfAttentionData_WithPastAndPresent_NoMask_NoRelPosBias.present_value_data", data.present_value_data);
  data.is_static_kv = false;
}

void GetSelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias(AttentionTestData& data) {
  data.hidden_size = 16;
  data.v_hidden_size = 16;
  data.num_heads = 2;
  data.batch_size = 2;
  data.sequence_length = 1;
  data.kv_sequence_length = 1;
  data.mask_type = AttentionMaskType::MASK_NONE;

  data.skip_kernel_types = {
      AttentionKernelType::AttentionKernel_TrtFlashAttention,
      AttentionKernelType::AttentionKernel_TrtFusedCrossAttention,
      AttentionKernelType::AttentionKernel_TrtFusedAttention,
      AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention,
  };

  LoadTensor("SelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias.query_data", data.query_data);
  LoadTensor("SelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias.key_data", data.key_data);
  LoadTensor("SelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias.value_data", data.value_data);
  LoadTensor("SelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias.bias_data", data.bias_data);
  LoadTensor("SelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias.past_key_data",
             data.past_key_data);
  LoadTensor("SelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias.past_value_data",
             data.past_value_data);
  LoadTensor("SelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias.fp32_output_data",
             data.fp32_output_data);
  data.fp16_output_data = data.fp32_output_data;
  LoadTensor("SelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias.present_key_data",
             data.present_key_data);
  LoadTensor("SelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias.present_value_data",
             data.present_value_data);
  data.is_static_kv = false;
}

void GetSelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias_NoBias(AttentionTestData& data) {
  GetSelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias(data);
  data.bias_data.clear();
  LoadTensor("SelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias_NoBias.past_key_data",
             data.past_key_data);
  LoadTensor("SelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias_NoBias.past_value_data",
             data.past_value_data);
  LoadTensor("SelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias_NoBias.fp32_output_data",
             data.fp32_output_data);
  data.fp16_output_data = data.fp32_output_data;
  LoadTensor("SelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias_NoBias.present_key_data",
             data.present_key_data);
  LoadTensor("SelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias_NoBias.present_value_data",
             data.present_value_data);
  data.is_static_kv = false;
}

void GetCrossAttentionData_WithPastPassedInDirectly_NoMask(AttentionTestData& data) {
  data.hidden_size = 4;
  data.v_hidden_size = 4;
  data.num_heads = 2;
  data.batch_size = 2;
  data.sequence_length = 2;
  data.kv_sequence_length = 3;
  data.mask_type = AttentionMaskType::MASK_NONE;

  data.skip_kernel_types = {
      AttentionKernelType::AttentionKernel_TrtFlashAttention,
      AttentionKernelType::AttentionKernel_TrtFusedCrossAttention,
      AttentionKernelType::AttentionKernel_TrtFusedAttention,
      AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention,
  };

  LoadTensor("CrossAttentionData_WithPastPassedInDirectly_NoMask.query_data", data.query_data);
  LoadTensor("CrossAttentionData_WithPastPassedInDirectly_NoMask.past_key_data", data.past_key_data);
  LoadTensor("CrossAttentionData_WithPastPassedInDirectly_NoMask.past_value_data", data.past_value_data);
  LoadTensor("CrossAttentionData_WithPastPassedInDirectly_NoMask.fp32_output_data", data.fp32_output_data);
  data.fp16_output_data = data.fp32_output_data;
}

void GetCausal_EmptyPastState(std::vector<float>& input, std::vector<float>& output, std::vector<float>& present) {
  LoadTensor("Causal_EmptyPastState.input_data", input);
  LoadTensor("Causal_EmptyPastState.output_data", output);
  LoadTensor("Causal_EmptyPastState.present_data", present);
}

void GetAttentionDataWithNeoXRotaryEmbedding(std::vector<float>& input,
                                             std::vector<float>& weights,
                                             std::vector<float>& bias,
                                             std::vector<float>& output) {
  LoadTensor("AttentionDataWithNeoXRotaryEmbedding.input", input);
  LoadTensor("AttentionDataWithNeoXRotaryEmbedding.weights", weights);
  LoadTensor("AttentionDataWithNeoXRotaryEmbedding.bias", bias);
  LoadTensor("AttentionDataWithNeoXRotaryEmbedding.output", output);
}

void GetPackedMultiHeadAttentionData_Batch2_HeadSize32_NoRelPosBias(PackedAttentionTestData& data) {
  data.hidden_size = 32;
  data.v_hidden_size = 32;
  data.num_heads = 1;
  data.batch_size = 2;
  data.sequence_length = 2;
  data.kv_sequence_length = 2;

  data.token_offset = {0, 2, 3, 1};
  data.cumulative_sequence_length = {0, 1, 3};
  data.token_count = 3;

  data.skip_kernel_types = {
      AttentionKernelType::AttentionKernel_TrtFusedCrossAttention};

  LoadTensor("PackedMultiHeadAttentionData_Batch2_HeadSize32_NoRelPosBias.query_data", data.query_data);
  LoadTensor("PackedMultiHeadAttentionData_Batch2_HeadSize32_NoRelPosBias.key_data", data.key_data);
  LoadTensor("PackedMultiHeadAttentionData_Batch2_HeadSize32_NoRelPosBias.value_data", data.value_data);
  data.bias_data = {};
  LoadTensor("PackedMultiHeadAttentionData_Batch2_HeadSize32_NoRelPosBias.qkv_data", data.qkv_data);

  // Do not test fp32
  data.fp32_output_data = {};

  LoadTensor("PackedMultiHeadAttentionData_Batch2_HeadSize32_NoRelPosBias.fp16_output_data", data.fp16_output_data);
}

void GetPackedMultiHeadAttentionData_Batch2_HeadSize8_RelPosBias(PackedAttentionTestData& data) {
  data.hidden_size = 16;
  data.v_hidden_size = 16;
  data.num_heads = 2;
  data.batch_size = 2;
  data.sequence_length = 2;
  data.kv_sequence_length = 2;
  data.token_offset = {0, 2, 3, 1};
  data.cumulative_sequence_length = {0, 1, 3};
  data.token_count = 3;

  data.skip_kernel_types = {
      AttentionKernelType::AttentionKernel_TrtFusedCrossAttention};

  LoadTensor("PackedMultiHeadAttentionData_Batch2_HeadSize8_RelPosBias.query_data", data.query_data);
  LoadTensor("PackedMultiHeadAttentionData_Batch2_HeadSize8_RelPosBias.key_data", data.key_data);
  LoadTensor("PackedMultiHeadAttentionData_Batch2_HeadSize8_RelPosBias.value_data", data.value_data);
  data.bias_data = {};
  LoadTensor("PackedMultiHeadAttentionData_Batch2_HeadSize8_RelPosBias.qkv_data", data.qkv_data);

  // shape: batch_size, num_heads, sequence_length, sequence_length
  LoadTensor("PackedMultiHeadAttentionData_Batch2_HeadSize8_RelPosBias.rel_pos_bias_data", data.rel_pos_bias_data);
  data.broadcast_rel_pos_bias = false;

  // Do not test fp32
  data.fp32_output_data = {};

  LoadTensor("PackedMultiHeadAttentionData_Batch2_HeadSize8_RelPosBias.fp16_output_data", data.fp16_output_data);
}

void GetPackedMultiHeadAttentionData_Batch2_HeadSize8_BroadcastRelPosBias(PackedAttentionTestData& data) {
  data.hidden_size = 16;
  data.v_hidden_size = 16;
  data.num_heads = 2;
  data.batch_size = 2;
  data.sequence_length = 8;
  data.kv_sequence_length = 8;
  data.token_offset = {0, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7};
  data.cumulative_sequence_length = {0, 1, 9};
  data.token_count = 9;

  data.skip_kernel_types = {
      AttentionKernelType::AttentionKernel_TrtFusedCrossAttention};

  LoadTensor("PackedMultiHeadAttentionData_Batch2_HeadSize8_BroadcastRelPosBias.query_data", data.query_data);
  LoadTensor("PackedMultiHeadAttentionData_Batch2_HeadSize8_BroadcastRelPosBias.key_data", data.key_data);
  LoadTensor("PackedMultiHeadAttentionData_Batch2_HeadSize8_BroadcastRelPosBias.value_data", data.value_data);
  data.bias_data = {};
  LoadTensor("PackedMultiHeadAttentionData_Batch2_HeadSize8_BroadcastRelPosBias.qkv_data", data.qkv_data);

  // shape: 1, num_heads, sequence_length, sequence_length
  LoadTensor("PackedMultiHeadAttentionData_Batch2_HeadSize8_BroadcastRelPosBias.rel_pos_bias_data",
             data.rel_pos_bias_data);
  data.broadcast_rel_pos_bias = true;

  // Do not test fp32
  data.fp32_output_data = {};

  LoadTensor("PackedMultiHeadAttentionData_Batch2_HeadSize8_BroadcastRelPosBias.fp16_output_data",
             data.fp16_output_data);
}

bool SkipAttentionKernel(AttentionTestData& data, AttentionKernelType kernel_type) {
  return data.skip_kernel_types.end() != std::find(data.skip_kernel_types.begin(),
                                                   data.skip_kernel_types.end(),
                                                   kernel_type);
}

}  // namespace test
}  // namespace onnxruntime
