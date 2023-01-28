// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/env_var_utils.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/contrib_ops/attention_op_test_helper.h"

namespace onnxruntime {
namespace test {

static void RunMultiHeadAttentionTest(
    const std::vector<float>& query_data,               // query:  [batch_size, sequence_length, hidden_size]
    const std::vector<float>& key_data,                 // key:    [batch_size, kv_sequence_length, hidden_size]
    const std::vector<float>& value_data,               // value:  [batch_size, kv_sequence_length, v_hidden_size]
    const std::vector<float>& bias_data,                // bias:   [hidden_size + hidden_size + v_hidden_size]
    const std::vector<int32_t>& key_padding_mask_data,  // key_padding_mask: see below
    AttentionMaskType mask_type,                        // 1 for [batch_size], 2 for [batch_size, kv_sequence_length]
    const std::vector<float>& output_data,              // output: [batch_size, sequence_length, v_hidden_size]
    int num_heads,
    int batch_size,
    int sequence_length,
    int kv_sequence_length,
    int hidden_size,
    int v_hidden_size,
    bool use_float16 = false,
    bool disable_cpu = true,  // not supported in cpu right now.
    bool disable_cuda = false,
    bool disable_rocm = true)  // not supported in rocm right now.
{
  kv_sequence_length = (kv_sequence_length == 0 ? sequence_length : kv_sequence_length);

  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture) && !disable_cuda;
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get()) && !disable_rocm;
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get()) && !use_float16 && !disable_cpu;

  if (enable_cpu || enable_cuda || enable_rocm) {
    OpTester tester("MultiHeadAttention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
    tester.AddAttribute<float>("mask_filter_value", static_cast<float>(-10000.0f));

    std::vector<int64_t> query_dims = {batch_size, sequence_length, hidden_size};
    std::vector<int64_t> key_dims = {batch_size, kv_sequence_length, hidden_size};
    std::vector<int64_t> value_dims = {batch_size, kv_sequence_length, v_hidden_size};
    std::vector<int64_t> bias_dims = {hidden_size + hidden_size + v_hidden_size};
    std::vector<int64_t> output_dims = {batch_size, sequence_length, v_hidden_size};

    std::vector<int64_t> mask_dims_1 = {batch_size};
    std::vector<int64_t> mask_dims_2 = {batch_size, kv_sequence_length};
    std::vector<int64_t>& key_padding_mask_dims = (mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN)
                                                      ? mask_dims_1
                                                      : mask_dims_2;

    if (use_float16) {
      tester.AddInput<MLFloat16>("query", query_dims, ToFloat16(query_data));
      tester.AddInput<MLFloat16>("key", key_dims, ToFloat16(key_data));
      tester.AddInput<MLFloat16>("value", value_dims, ToFloat16(value_data));

      if (bias_data.size()) {
        tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
      } else {
        tester.AddOptionalInputEdge<MLFloat16>();
      }

      if (key_padding_mask_data.size()) {
        tester.AddInput<int32_t>("key_padding_mask", key_padding_mask_dims, key_padding_mask_data);
      } else {
        tester.AddOptionalInputEdge<int32_t>();
      }

      constexpr float rel_error = 0.0f;
      constexpr float abs_error = 0.05f;
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data), /*sort*/ false, rel_error, abs_error);
    } else {
      tester.AddInput<float>("query", query_dims, query_data);
      tester.AddInput<float>("key", key_dims, key_data);
      tester.AddInput<float>("value", value_dims, value_data);

      if (bias_data.size()) {
        tester.AddInput<float>("bias", bias_dims, bias_data);
      } else {
        tester.AddOptionalInputEdge<float>();
      }

      if (key_padding_mask_data.size()) {
        tester.AddInput<int32_t>("key_padding_mask", key_padding_mask_dims, key_padding_mask_data);
      } else {
        tester.AddOptionalInputEdge<int32_t>();
      }

      constexpr float rel_error = 0.0f;
      constexpr float abs_error = 0.02f;
      tester.AddOutput<float>("output", output_dims, output_data, /*sort*/ false, rel_error, abs_error);
    }

    if (enable_cuda) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_rocm) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultRocmExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_cpu) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCpuExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}

static void RunMultiHeadAttentionKernel(
    const std::vector<float>& query_data,               // query:  [batch_size, sequence_length, hidden_size]
    const std::vector<float>& key_data,                 // key:    [batch_size, kv_sequence_length, hidden_size]
    const std::vector<float>& value_data,               // value:  [batch_size, kv_sequence_length, v_hidden_size]
    const std::vector<float>& bias_data,                // bias:   [hidden_size + hidden_size + v_hidden_size]
    const std::vector<int32_t>& key_padding_mask_data,  // key_padding_mask: see below
    AttentionMaskType mask_type,                        // 1 for [batch_size], 2 for [batch_size, kv_sequence_length]
    const std::vector<float>& output_data,              // output: [batch_size, sequence_length, v_hidden_size]
    int num_heads,
    int batch_size,
    int sequence_length,
    int kv_sequence_length,
    int hidden_size,
    int v_hidden_size,
    AttentionKernelType kernel_type,
    bool use_float16 = true,
    bool disable_cpu = true,  // not supported in cpu right now.
    bool disable_cuda = false,
    bool disable_rocm = true) {
  if (kernel_type == AttentionKernelType::AttentionKernel_Default) {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "0"},
            {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "0"}}};
    RunMultiHeadAttentionTest(
        query_data, key_data, value_data, bias_data, key_padding_mask_data, mask_type, output_data,
        num_heads, batch_size, sequence_length, kv_sequence_length, hidden_size, v_hidden_size,
        use_float16, disable_cpu, disable_cuda, disable_rocm);
    return;
  }

  if (kernel_type == AttentionKernelType::AttentionKernel_Unfused) {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
            {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"}}};
    RunMultiHeadAttentionTest(
        query_data, key_data, value_data, bias_data, key_padding_mask_data, mask_type, output_data,
        num_heads, batch_size, sequence_length, kv_sequence_length, hidden_size, v_hidden_size,
        use_float16, disable_cpu, disable_cuda, disable_rocm);
    return;
  }

  if (kernel_type == AttentionKernelType::AttentionKernel_TrtFusedCrossAttention) {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "0"},
            {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"}}};
    RunMultiHeadAttentionTest(
        query_data, key_data, value_data, bias_data, key_padding_mask_data, mask_type, output_data,
        num_heads, batch_size, sequence_length, kv_sequence_length, hidden_size, v_hidden_size,
        use_float16, disable_cpu, disable_cuda, disable_rocm);
    return;
  }

#if USE_FLASH_ATTENTION
  if (kernel_type == AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention) {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
            {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "0"}}};
    RunMultiHeadAttentionTest(
        query_data, key_data, value_data, bias_data, key_padding_mask_data, mask_type, output_data,
        num_heads, batch_size, sequence_length, kv_sequence_length, hidden_size, v_hidden_size,
        use_float16, disable_cpu, disable_cuda, disable_rocm);
    return;
  }
#endif

  if (kernel_type == AttentionKernelType::AttentionKernel_TrtFusedAttention) {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
            {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"}}};
    RunMultiHeadAttentionTest(
        query_data, key_data, value_data, bias_data, key_padding_mask_data, mask_type, output_data,
        num_heads, batch_size, sequence_length, kv_sequence_length, hidden_size, v_hidden_size,
        use_float16, disable_cpu, disable_cuda, disable_rocm);
  }
}

static void RunMultiHeadAttentionTests(AttentionTestData& data) {
  if (data.fp32_output_data.size() > 0) {
    constexpr bool use_float16 = false;

    AttentionKernelType kernel_type = AttentionKernelType::AttentionKernel_Unfused;
    if (!SkipAttentionKernel(data, kernel_type)) {
      RunMultiHeadAttentionKernel(
          data.query_data, data.key_data, data.value_data, data.bias_data, data.key_padding_mask_data, data.mask_type,
          data.fp32_output_data, data.num_heads, data.batch_size, data.sequence_length, data.kv_sequence_length,
          data.hidden_size, data.v_hidden_size, kernel_type, use_float16);
    }

#if USE_FLASH_ATTENTION
    if (data.sequence_length >= contrib::attention::kMinSequenceLengthForMemoryEfficientAttentionFp32 ||
        data.kv_sequence_length >= contrib::attention::kMinSequenceLengthForMemoryEfficientAttentionFp32) {
      kernel_type = AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention;
      if (!SkipAttentionKernel(data, kernel_type)) {
        RunMultiHeadAttentionKernel(
            data.query_data, data.key_data, data.value_data, data.bias_data, data.key_padding_mask_data, data.mask_type,
            data.fp32_output_data, data.num_heads, data.batch_size, data.sequence_length, data.kv_sequence_length,
            data.hidden_size, data.v_hidden_size, kernel_type, use_float16);
      }
    }
#endif

    kernel_type = AttentionKernelType::AttentionKernel_Default;
    RunMultiHeadAttentionKernel(
        data.query_data, data.key_data, data.value_data, data.bias_data, data.key_padding_mask_data, data.mask_type,
        data.fp32_output_data, data.num_heads, data.batch_size, data.sequence_length, data.kv_sequence_length,
        data.hidden_size, data.v_hidden_size, kernel_type, use_float16);
  }

  if (data.fp16_output_data.size() > 0) {
    constexpr bool use_float16 = true;
    AttentionKernelType kernel_type = AttentionKernelType::AttentionKernel_TrtFusedCrossAttention;
    if (!SkipAttentionKernel(data, kernel_type)) {
      RunMultiHeadAttentionKernel(
          data.query_data, data.key_data, data.value_data, data.bias_data, data.key_padding_mask_data, data.mask_type,
          data.fp16_output_data, data.num_heads, data.batch_size, data.sequence_length, data.kv_sequence_length,
          data.hidden_size, data.v_hidden_size, kernel_type, use_float16);
    }

    kernel_type = AttentionKernelType::AttentionKernel_TrtFusedAttention;
    if (!SkipAttentionKernel(data, kernel_type)) {
      RunMultiHeadAttentionKernel(
          data.query_data, data.key_data, data.value_data, data.bias_data, data.key_padding_mask_data, data.mask_type,
          data.fp16_output_data, data.num_heads, data.batch_size, data.sequence_length, data.kv_sequence_length,
          data.hidden_size, data.v_hidden_size, kernel_type, use_float16);
    }

#if USE_FLASH_ATTENTION
    kernel_type = AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention;
    if (!SkipAttentionKernel(data, kernel_type)) {
      RunMultiHeadAttentionKernel(
          data.query_data, data.key_data, data.value_data, data.bias_data, data.key_padding_mask_data, data.mask_type,
          data.fp16_output_data, data.num_heads, data.batch_size, data.sequence_length, data.kv_sequence_length,
          data.hidden_size, data.v_hidden_size, kernel_type, use_float16);
    }
#endif

    kernel_type = AttentionKernelType::AttentionKernel_Default;
    RunMultiHeadAttentionKernel(
        data.query_data, data.key_data, data.value_data, data.bias_data, data.key_padding_mask_data, data.mask_type,
        data.fp16_output_data, data.num_heads, data.batch_size, data.sequence_length, data.kv_sequence_length,
        data.hidden_size, data.v_hidden_size, kernel_type, use_float16);
  }
}

#ifndef _MSC_VER
// Test fused cross attention kernel
// It requires head_size > 32 and head_size <= 64 for T4 GPU; hidden_size == v_hidden_size.
TEST(MultiHeadAttentionTest, CrossAttention_Batch2_HeadSize40) {
  AttentionTestData data;
  GetCrossAttentionData_HeadSize40(data);
  RunMultiHeadAttentionTests(data);
}

TEST(MultiHeadAttentionTest, CrossAttention_Batch2_HeadSize32_RightSidePadding_Mask1D) {
  AttentionTestData data;
  GetCrossAttentionData_Batch2_HeadSize32_RightSidePadding(data, true);
  RunMultiHeadAttentionTests(data);
}

TEST(MultiHeadAttentionTest, CrossAttention_Batch2_HeadSize32_RightSidePadding_Mask2D) {
  AttentionTestData data;
  GetCrossAttentionData_Batch2_HeadSize32_RightSidePadding(data, false);
  RunMultiHeadAttentionTests(data);
}

TEST(MultiHeadAttentionTest, CrossAttention_Batch1_HeadSize32_LeftSidePadding_Mask2D) {
  AttentionTestData data;
  GetCrossAttentionData_Batch1_HeadSize32_LeftSidePadding(data);
  RunMultiHeadAttentionTests(data);
}
#endif

// This tests qk_head_size != k_head_size
TEST(MultiHeadAttentionTest, CrossAttention_Batch2_HeadSize16_8) {
  AttentionTestData data;
  GetCrossAttentionData_HeadSize16_8(data);
  RunMultiHeadAttentionTests(data);
}

TEST(MultiHeadAttentionTest, CrossAttention_Batch1_HeadSize16) {
  AttentionTestData data;
  GetCrossAttentionData_HeadSize16(data);
  RunMultiHeadAttentionTests(data);
}

}  // namespace test
}  // namespace onnxruntime
