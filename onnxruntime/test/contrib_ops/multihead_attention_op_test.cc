// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/env_var_utils.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/contrib_ops/attention_op_test_helper.h"

#if defined(USE_ROCM) && defined(USE_COMPOSABLE_KERNEL) && !defined(USE_MIGRAPHX)
#define DISABLE_ROCM false
#else
#define DISABLE_ROCM true
#endif

#if defined(USE_ROCM)
#define ROCM_GTEST_SKIP(message) GTEST_SKIP_(message)
#else
#define ROCM_GTEST_SKIP(message)
#endif

namespace onnxruntime {
namespace test {

static void RunMultiHeadAttentionTest(
    const std::vector<float>& query_data,               // query:  [batch_size, sequence_length, hidden_size]
    const std::vector<float>& key_data,                 // key:    [batch_size, kv_sequence_length, hidden_size]
    const std::vector<float>& value_data,               // value:  [batch_size, kv_sequence_length, v_hidden_size]
    const std::vector<float>& kv_data,                  // packed_kv:  [batch_size, kv_sequence_length, num_heads, 2, head_size]
    const std::vector<float>& qkv_data,                 // packed_qkv:  [batch_size, sequence_length, num_heads, 3, head_size]
    const std::vector<float>& bias_data,                // bias:   [hidden_size + hidden_size + v_hidden_size] or empty
    const std::vector<float>& rel_pos_bias_data,        // relative_position_bias: [1, num_heads, sequence_length, total_sequence_length]
    const std::vector<float>& past_key_data,            // past_key: [batch_size, num_heads, kv_sequence_length, head_size]
    const std::vector<float>& past_value_data,          // past_value: [batch_size, num_heads, kv_sequence_length, head_size]
    const std::vector<float>& present_key_data,         // present_key: [batch_size, num_heads, total_sequence_length, head_size]
    const std::vector<float>& present_value_data,       // present_value: [batch_size, num_heads, total_sequence_length, head_size]
    const std::vector<int32_t>& key_padding_mask_data,  // key_padding_mask: see below
    AttentionMaskType mask_type,                        // 1 for [batch_size], 2 for [batch_size, kv_sequence_length]
    const std::vector<float>& output_data,              // output: [batch_size, sequence_length, v_hidden_size]
    int num_heads,
    int batch_size,
    int sequence_length,
    int kv_sequence_length,
    int hidden_size,
    int v_hidden_size,
    bool is_static_kv = true,
    bool use_float16 = false,
    bool disable_cpu = false,  // some cases not supported in cpu right now.
    bool disable_cuda = false,
    bool disable_rocm = DISABLE_ROCM,  // not supported in rocm right now.
    bool disable_dml = false) {
  kv_sequence_length = (kv_sequence_length == 0 ? sequence_length : kv_sequence_length);

  int min_cuda_architecture = use_float16 ? 750 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture) && !disable_cuda;
  // rocm mha is required to work with TunableOp Enabled
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider(/*test_tunable_op=*/true).get()) && !disable_rocm;
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get()) && !use_float16 && !disable_cpu;
  bool enable_dml = (nullptr != DefaultDmlExecutionProvider().get()) && !disable_dml;

  if (enable_rocm && !use_float16) {
    LOGS_DEFAULT(WARNING) << "ROCm MHA only have kernel for half datatype implemented, skip float datatype tests";
    enable_rocm = false;
  }

  if (enable_rocm && !bias_data.empty()) {
    LOGS_DEFAULT(WARNING) << "ROCm MHA does not support qkv_bias, skip qkv_bias tests";
    enable_rocm = false;
  }

  if (enable_cpu || enable_cuda || enable_rocm || enable_dml) {
    OpTester tester("MultiHeadAttention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
    tester.AddAttribute<float>("mask_filter_value", static_cast<float>(-10000.0f));

    std::vector<int64_t> query_dims = {batch_size, sequence_length, hidden_size};
    std::vector<int64_t> key_dims = {batch_size, is_static_kv ? kv_sequence_length : sequence_length, hidden_size};
    std::vector<int64_t> value_dims = {batch_size, is_static_kv ? kv_sequence_length : sequence_length, v_hidden_size};
    std::vector<int64_t> bias_dims = {hidden_size + hidden_size + v_hidden_size};
    // TODO(wy): Introduce past sequence length to avoid using kv_sequence_length.
    std::vector<int64_t> rel_pos_bias_dims =
        {1, num_heads, sequence_length, past_key_data.size() ? sequence_length + kv_sequence_length : sequence_length};
    std::vector<int64_t> past_key_dims = {batch_size, num_heads, kv_sequence_length, hidden_size / num_heads};
    std::vector<int64_t> past_value_dims = past_key_dims;
    std::vector<int64_t> output_dims = {batch_size, sequence_length, v_hidden_size};
    std::vector<int64_t> present_key_dims =
        {batch_size, num_heads, is_static_kv ? kv_sequence_length : sequence_length + kv_sequence_length, hidden_size / num_heads};
    std::vector<int64_t> present_value_dims = present_key_dims;

    std::vector<float> query = (qkv_data.size() > 0 ? qkv_data : query_data);
    std::vector<float> key;
    std::vector<float> value;
    if (qkv_data.size() == 0) {
      if (kv_data.size() > 0) {
        ORT_ENFORCE(hidden_size == v_hidden_size);
        key = kv_data;
        key_dims = {batch_size, kv_sequence_length, num_heads, 2, hidden_size / num_heads};
      } else {
        key = key_data;
        value = value_data;
      }
    } else {
      ORT_ENFORCE(sequence_length == kv_sequence_length && hidden_size == v_hidden_size);
      query_dims = {batch_size, sequence_length, num_heads, 3, hidden_size / num_heads};
    }

    std::vector<int64_t> mask_dims_1 = {batch_size};
    std::vector<int64_t> mask_dims_2 = {batch_size, kv_sequence_length};
    std::vector<int64_t> mask_dims_3 = {3 * batch_size + 2};
    std::vector<int64_t>& key_padding_mask_dims = (mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN)
                                                      ? mask_dims_1
                                                      : (mask_type == AttentionMaskType::MASK_2D_KEY_PADDING ? mask_dims_2 : mask_dims_3);

    if (use_float16) {
      tester.AddInput<MLFloat16>("query", query_dims, ToFloat16(query));

      if (key.size()) {
        tester.AddInput<MLFloat16>("key", key_dims, ToFloat16(key));
      } else if (past_key_data.size() && is_static_kv == true) {
        tester.AddInput<MLFloat16>("key", past_key_dims, ToFloat16(past_key_data));
      } else {
        tester.AddOptionalInputEdge<MLFloat16>();
      }

      if (value.size()) {
        tester.AddInput<MLFloat16>("value", value_dims, ToFloat16(value));
      } else if (past_value_data.size() && is_static_kv == true) {
        tester.AddInput<MLFloat16>("value", past_value_dims, ToFloat16(past_value_data));
      } else {
        tester.AddOptionalInputEdge<MLFloat16>();
      }

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

      if (rel_pos_bias_data.size()) {
        tester.AddInput<MLFloat16>("relative_position_bias", rel_pos_bias_dims, ToFloat16(rel_pos_bias_data));
      } else {
        tester.AddOptionalInputEdge<MLFloat16>();
      }

      if (past_key_data.size() && is_static_kv == false) {
        tester.AddInput<MLFloat16>("past_key", past_key_dims, ToFloat16(past_key_data));
      } else {
        tester.AddOptionalInputEdge<MLFloat16>();
      }

      if (past_value_data.size() && is_static_kv == false) {
        tester.AddInput<MLFloat16>("past_value", past_value_dims, ToFloat16(past_value_data));
      } else {
        tester.AddOptionalInputEdge<MLFloat16>();
      }

      constexpr float rel_error = 0.0f;
      constexpr float abs_error = 0.05f;
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data), /*sort*/ false, rel_error, abs_error);

      if (present_key_data.size()) {
        tester.AddOutput<MLFloat16>("present_key", present_key_dims, ToFloat16(present_key_data), /*sort*/ false, rel_error, abs_error);
      } else {
        tester.AddOptionalOutputEdge<MLFloat16>();
      }

      if (present_value_data.size()) {
        tester.AddOutput<MLFloat16>("present_value", present_value_dims, ToFloat16(present_value_data), /*sort*/ false, rel_error, abs_error);
      } else {
        tester.AddOptionalOutputEdge<MLFloat16>();
      }
    } else {
      tester.AddInput<float>("query", query_dims, query);

      if (key.size()) {
        tester.AddInput<float>("key", key_dims, key);
      } else if (past_key_data.size() && is_static_kv == true) {
        tester.AddInput<float>("key", past_key_dims, past_key_data);
      } else {
        tester.AddOptionalInputEdge<float>();
      }

      if (value.size()) {
        tester.AddInput<float>("value", value_dims, value);
      } else if (past_value_data.size() && is_static_kv == true) {
        tester.AddInput<float>("value", past_value_dims, past_value_data);
      } else {
        tester.AddOptionalInputEdge<float>();
      }

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

      if (rel_pos_bias_data.size()) {
        tester.AddInput<float>("relative_position_bias", rel_pos_bias_dims, rel_pos_bias_data);
      } else {
        tester.AddOptionalInputEdge<float>();
      }

      if (past_key_data.size() && is_static_kv == false) {
        tester.AddInput<float>("past_key", past_key_dims, past_key_data);
      } else {
        tester.AddOptionalInputEdge<float>();
      }

      if (past_value_data.size() && is_static_kv == false) {
        tester.AddInput<float>("past_value", past_value_dims, past_value_data);
      } else {
        tester.AddOptionalInputEdge<float>();
      }

      constexpr float rel_error = 0.0f;
      constexpr float abs_error = 0.02f;
      tester.AddOutput<float>("output", output_dims, output_data, /*sort*/ false, rel_error, abs_error);

      if (present_key_data.size()) {
        tester.AddOutput<float>("present_key", present_key_dims, present_key_data, /*sort*/ false, rel_error, abs_error);
      } else {
        tester.AddOptionalOutputEdge<float>();
      }

      if (present_value_data.size()) {
        tester.AddOutput<float>("present_value", present_value_dims, present_value_data, /*sort*/ false, rel_error, abs_error);
      } else {
        tester.AddOptionalOutputEdge<float>();
      }
    }

    if (enable_cuda) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_rocm) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultRocmExecutionProvider(/*test_tunable_op=*/true));
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_cpu) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCpuExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_dml) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultDmlExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}

static void RunMultiHeadAttentionKernel(
    const std::vector<float>& query_data,               // query:  [batch_size, sequence_length, hidden_size]
    const std::vector<float>& key_data,                 // key:    [batch_size, kv_sequence_length, hidden_size]
    const std::vector<float>& value_data,               // value:  [batch_size, kv_sequence_length, v_hidden_size]
    const std::vector<float>& kv_data,                  // packed_kv:  [batch_size, kv_sequence_length, num_heads, 2, head_size]
    const std::vector<float>& qkv_data,                 // packed_qkv:  [batch_size, sequence_length, num_heads, 3, head_size]
    const std::vector<float>& bias_data,                // bias:   [hidden_size + hidden_size + v_hidden_size]
    const std::vector<float>& rel_pos_bias_data,        // relative_position_bias: [1, num_heads, sequence_length, total_sequence_length]
    const std::vector<float>& past_key_data,            // past_key: [batch_size, num_heads, kv_sequence_length, head_size]
    const std::vector<float>& past_value_data,          // past_value: [batch_size, num_heads, kv_sequence_length, head_size]
    const std::vector<float>& present_key_data,         // present_key: [batch_size, num_heads, total_sequence_length, head_size]
    const std::vector<float>& present_value_data,       // present_value: [batch_size, num_heads, total_sequence_length, head_size]
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
    bool is_static_kv = true,
    bool disable_cpu = false,  // some cases not supported in cpu right now.
    bool disable_cuda = false,
    bool disable_rocm = DISABLE_ROCM,
    bool disable_dml = false) {
  if (kernel_type == AttentionKernelType::AttentionKernel_Default) {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "0"},
            {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "0"}}};
    RunMultiHeadAttentionTest(
        query_data, key_data, value_data, kv_data, qkv_data, bias_data, rel_pos_bias_data,
        past_key_data, past_value_data, present_key_data, present_value_data, key_padding_mask_data,
        mask_type, output_data, num_heads, batch_size, sequence_length, kv_sequence_length,
        hidden_size, v_hidden_size, is_static_kv, use_float16, disable_cpu, disable_cuda, disable_rocm, disable_dml);
    return;
  }

  if (kernel_type == AttentionKernelType::AttentionKernel_Unfused) {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
            {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"}}};
    RunMultiHeadAttentionTest(
        query_data, key_data, value_data, kv_data, qkv_data, bias_data, rel_pos_bias_data,
        past_key_data, past_value_data, present_key_data, present_value_data, key_padding_mask_data,
        mask_type, output_data, num_heads, batch_size, sequence_length, kv_sequence_length,
        hidden_size, v_hidden_size, is_static_kv, use_float16, disable_cpu, disable_cuda, disable_rocm, disable_dml);
    return;
  }

  if (kernel_type == AttentionKernelType::AttentionKernel_TrtFusedCrossAttention) {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "0"},
            {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"}}};
    RunMultiHeadAttentionTest(
        query_data, key_data, value_data, kv_data, qkv_data, bias_data, rel_pos_bias_data,
        past_key_data, past_value_data, present_key_data, present_value_data, key_padding_mask_data,
        mask_type, output_data, num_heads, batch_size, sequence_length, kv_sequence_length,
        hidden_size, v_hidden_size, is_static_kv, use_float16, disable_cpu, disable_cuda, disable_rocm, disable_dml);
    return;
  }

#if USE_FLASH_ATTENTION
  if (kernel_type == AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention) {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
            {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "0"}}};
    RunMultiHeadAttentionTest(
        query_data, key_data, value_data, kv_data, qkv_data, bias_data, rel_pos_bias_data,
        past_key_data, past_value_data, present_key_data, present_value_data, key_padding_mask_data,
        mask_type, output_data, num_heads, batch_size, sequence_length, kv_sequence_length,
        hidden_size, v_hidden_size, is_static_kv, use_float16, disable_cpu, disable_cuda, disable_rocm, disable_dml);
    return;
  }
#endif

  if (kernel_type == AttentionKernelType::AttentionKernel_TrtFusedAttention) {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
            {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"}}};
    RunMultiHeadAttentionTest(
        query_data, key_data, value_data, kv_data, qkv_data, bias_data, rel_pos_bias_data,
        past_key_data, past_value_data, present_key_data, present_value_data, key_padding_mask_data,
        mask_type, output_data, num_heads, batch_size, sequence_length, kv_sequence_length,
        hidden_size, v_hidden_size, is_static_kv, use_float16, disable_cpu, disable_cuda, disable_rocm, disable_dml);
  }
}

static void RunMultiHeadAttentionTests(AttentionTestData& data, bool disable_cpu = false, bool disable_cuda = false) {
  if (data.fp32_output_data.size() > 0) {
    constexpr bool use_float16 = false;

    AttentionKernelType kernel_type = AttentionKernelType::AttentionKernel_Unfused;
    if (!SkipAttentionKernel(data, kernel_type)) {
      RunMultiHeadAttentionKernel(
          data.query_data, data.key_data, data.value_data, data.kv_data, data.qkv_data, data.bias_data,
          data.rel_pos_bias_data, data.past_key_data, data.past_value_data, data.present_key_data,
          data.present_value_data, data.key_padding_mask_data, data.mask_type, data.fp32_output_data,
          data.num_heads, data.batch_size, data.sequence_length, data.kv_sequence_length, data.hidden_size,
          data.v_hidden_size, kernel_type, use_float16, data.is_static_kv, disable_cpu, disable_cuda);
    }

#if USE_FLASH_ATTENTION
    if (data.sequence_length >= contrib::attention::kMinSequenceLengthForMemoryEfficientAttentionFp32 ||
        data.kv_sequence_length >= contrib::attention::kMinSequenceLengthForMemoryEfficientAttentionFp32) {
      kernel_type = AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention;
      if (!SkipAttentionKernel(data, kernel_type)) {
        RunMultiHeadAttentionKernel(
            data.query_data, data.key_data, data.value_data, data.kv_data, data.qkv_data, data.bias_data,
            data.rel_pos_bias_data, data.past_key_data, data.past_value_data, data.present_key_data,
            data.present_value_data, data.key_padding_mask_data, data.mask_type, data.fp32_output_data,
            data.num_heads, data.batch_size, data.sequence_length, data.kv_sequence_length, data.hidden_size,
            data.v_hidden_size, kernel_type, use_float16, data.is_static_kv, disable_cpu, disable_cuda);
      }
    }
#endif

    kernel_type = AttentionKernelType::AttentionKernel_Default;
    RunMultiHeadAttentionKernel(
        data.query_data, data.key_data, data.value_data, data.kv_data, data.qkv_data, data.bias_data,
        data.rel_pos_bias_data, data.past_key_data, data.past_value_data, data.present_key_data,
        data.present_value_data, data.key_padding_mask_data, data.mask_type, data.fp32_output_data,
        data.num_heads, data.batch_size, data.sequence_length, data.kv_sequence_length, data.hidden_size,
        data.v_hidden_size, kernel_type, use_float16, data.is_static_kv, disable_cpu, disable_cuda);
  }

  if (data.fp16_output_data.size() > 0) {
    constexpr bool use_float16 = true;
    AttentionKernelType kernel_type = AttentionKernelType::AttentionKernel_TrtFusedCrossAttention;
    if (!SkipAttentionKernel(data, kernel_type)) {
      RunMultiHeadAttentionKernel(
          data.query_data, data.key_data, data.value_data, data.kv_data, data.qkv_data, data.bias_data,
          data.rel_pos_bias_data, data.past_key_data, data.past_value_data, data.present_key_data,
          data.present_value_data, data.key_padding_mask_data, data.mask_type, data.fp16_output_data,
          data.num_heads, data.batch_size, data.sequence_length, data.kv_sequence_length, data.hidden_size,
          data.v_hidden_size, kernel_type, use_float16, data.is_static_kv, disable_cpu, disable_cuda);
    }

    kernel_type = AttentionKernelType::AttentionKernel_TrtFusedAttention;
    if (!SkipAttentionKernel(data, kernel_type)) {
      RunMultiHeadAttentionKernel(
          data.query_data, data.key_data, data.value_data, data.kv_data, data.qkv_data, data.bias_data,
          data.rel_pos_bias_data, data.past_key_data, data.past_value_data, data.present_key_data,
          data.present_value_data, data.key_padding_mask_data, data.mask_type, data.fp16_output_data,
          data.num_heads, data.batch_size, data.sequence_length, data.kv_sequence_length, data.hidden_size,
          data.v_hidden_size, kernel_type, use_float16, data.is_static_kv, disable_cpu, disable_cuda);
    }

#if USE_FLASH_ATTENTION
    kernel_type = AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention;
    if (!SkipAttentionKernel(data, kernel_type)) {
      RunMultiHeadAttentionKernel(
          data.query_data, data.key_data, data.value_data, data.kv_data, data.qkv_data, data.bias_data,
          data.rel_pos_bias_data, data.past_key_data, data.past_value_data, data.present_key_data,
          data.present_value_data, data.key_padding_mask_data, data.mask_type, data.fp16_output_data,
          data.num_heads, data.batch_size, data.sequence_length, data.kv_sequence_length, data.hidden_size,
          data.v_hidden_size, kernel_type, use_float16, data.is_static_kv, disable_cpu, disable_cuda);
    }
#endif

    kernel_type = AttentionKernelType::AttentionKernel_Default;
    RunMultiHeadAttentionKernel(
        data.query_data, data.key_data, data.value_data, data.kv_data, data.qkv_data, data.bias_data,
        data.rel_pos_bias_data, data.past_key_data, data.past_value_data, data.present_key_data,
        data.present_value_data, data.key_padding_mask_data, data.mask_type, data.fp16_output_data,
        data.num_heads, data.batch_size, data.sequence_length, data.kv_sequence_length, data.hidden_size,
        data.v_hidden_size, kernel_type, use_float16, data.is_static_kv, disable_cpu, disable_cuda);
  }
}

// Disable some tests in Windows since prefast build might crash with large test data.
#if !defined(_MSC_VER)
// Test fused cross attention kernel
// It requires head_size > 32 and head_size <= 64 for T4 GPU; hidden_size == v_hidden_size.
TEST(MultiHeadAttentionTest, CrossAttention_Batch2_HeadSize40) {
  AttentionTestData data;
  GetCrossAttentionData_HeadSize40(data);
  RunMultiHeadAttentionTests(data);

  GetCrossAttentionData_HeadSize40_NoBias(data);
  RunMultiHeadAttentionTests(data);
}

TEST(MultiHeadAttentionTest, CrossAttention_Batch2_HeadSize32_RightSidePadding_Mask1D) {
  ROCM_GTEST_SKIP("ROCm MHA does not support mask type of MASK_1D_KEY_SEQ_LEN");
  AttentionTestData data;
  GetCrossAttentionData_Batch2_HeadSize32_RightSidePadding(data, true);
  RunMultiHeadAttentionTests(data, true);

  GetCrossAttentionData_Batch2_HeadSize32_RightSidePadding_NoBias(data, true);
  RunMultiHeadAttentionTests(data, true);
}

TEST(MultiHeadAttentionTest, CrossAttention_Batch2_HeadSize32_RightSidePadding_Mask2D) {
  ROCM_GTEST_SKIP("ROCm MHA expect failure due to ck bug");
  AttentionTestData data;
  GetCrossAttentionData_Batch2_HeadSize32_RightSidePadding(data, false);
  RunMultiHeadAttentionTests(data, true);

  GetCrossAttentionData_Batch2_HeadSize32_RightSidePadding_NoBias(data, false);
  RunMultiHeadAttentionTests(data, true);
}

TEST(MultiHeadAttentionTest, CrossAttention_Batch1_HeadSize32_LeftSidePadding_Mask2D) {
  AttentionTestData data;
  GetCrossAttentionData_Batch1_HeadSize32_LeftSidePadding(data);
  RunMultiHeadAttentionTests(data, true);

  GetCrossAttentionData_Batch1_HeadSize32_LeftSidePadding_NoBias(data);
  RunMultiHeadAttentionTests(data, true);
}

TEST(MultiHeadAttentionTest, CrossAttention_Batch2_HeadSize32_NoBias_NoMask_PackedKV) {
  AttentionTestData data;
  GetCrossAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedKV(data);
  RunMultiHeadAttentionTests(data);
}

TEST(MultiHeadAttentionTest, SelfAttention_Batch2_HeadSize32_NoBias_NoMask_PackedQKV) {
  AttentionTestData data;
  GetSelfAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedQKV(data);
  RunMultiHeadAttentionTests(data);
}
#endif

// This tests qk_head_size != v_head_size
TEST(MultiHeadAttentionTest, CrossAttention_Batch2_HeadSize16_8) {
  AttentionTestData data;
  GetCrossAttentionData_HeadSize16_8(data);
  RunMultiHeadAttentionTests(data);

  GetCrossAttentionData_HeadSize16_8_NoBias(data);
  RunMultiHeadAttentionTests(data);
}

TEST(MultiHeadAttentionTest, CrossAttention_Batch1_HeadSize16) {
  AttentionTestData data;
  GetCrossAttentionData_HeadSize16(data);
  RunMultiHeadAttentionTests(data);

  GetCrossAttentionData_HeadSize16_NoBias(data);
  RunMultiHeadAttentionTests(data);
}

TEST(MultiHeadAttentionTest, CrossAttentionWithPast) {
  ROCM_GTEST_SKIP("ROCm MHA only support head_size >= 8");
  AttentionTestData data;
  GetCrossAttentionDataWithPast(data);
  RunMultiHeadAttentionTests(data);
}

TEST(MultiHeadAttentionTest, SelfAttention_WithPast_WithRelPosBias_ForT5) {
  ROCM_GTEST_SKIP("ROCm MHA only support head_size >= 8");
  AttentionTestData data;
  GetSelfAttentionData_WithPast_WithRelPosBias_ForT5(data);
  RunMultiHeadAttentionTests(data, true);
}

TEST(MultiHeadAttentionTest, AttentionCutlassRelPosBias) {
  // ROCM_GTEST_SKIP("ROCm does not support cutlass");
  AttentionTestData data;
  GetAttentionDataCutlassRelPosBias(data);
  RunMultiHeadAttentionTests(data);
}

TEST(MultiHeadAttentionTest, CrossAttention_DiffSequenceLengths) {
  // Whisper decoder cross attention without mask and different sequence lengths for Q and K/V
  AttentionTestData data;
  GetCrossAttentionData_DiffSequenceLengths(data);
  RunMultiHeadAttentionTests(data);

  GetCrossAttentionData_DiffSequenceLengths_HeadSize8(data);
  RunMultiHeadAttentionTests(data, /*disable_cpu=*/false, /*disable_cuda=*/true);

  GetCrossAttentionData_DiffSequenceLengths_HeadSize8_NoBias(data);
  RunMultiHeadAttentionTests(data, /*disable_cpu=*/false, /*disable_cuda=*/true);
}

TEST(MultiHeadAttentionTest, SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias) {
  // Whisper decoder self attention with past_kv and present_kv
  AttentionTestData data;
  GetSelfAttentionData_WithPastAndPresent_NoMask_NoRelPosBias(data);
  RunMultiHeadAttentionTests(data);

  GetSelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias(data);
  RunMultiHeadAttentionTests(data, /*disable_cpu=*/false, /*disable_cuda=*/true);

  GetSelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoRelPosBias_NoBias(data);
  RunMultiHeadAttentionTests(data, /*disable_cpu=*/false, /*disable_cuda=*/true);
}

TEST(MultiHeadAttentionTest, CrossAttention_WithPastPassedInDirectly_NoMask) {
  // Whisper decoder cross attention with past_kv in place of current KV and no present_kv
  AttentionTestData data;
  GetCrossAttentionData_WithPastPassedInDirectly_NoMask(data);
  RunMultiHeadAttentionTests(data);
}

}  // namespace test
}  // namespace onnxruntime
