// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <type_traits>

#include "core/common/narrow.h"
#include "core/graph/model.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/util/include/test_environment.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cuda/bert/packed_multihead_attention_impl.h"
#include "test/contrib_ops/attention_op_test_helper.h"

namespace onnxruntime {
using contrib::AttentionMaskType;
namespace test {

namespace {

class ScopedStdoutCapture {
 public:
  explicit ScopedStdoutCapture(bool enabled) : capturing_(enabled) {
    if (capturing_) {
      testing::internal::CaptureStdout();
    }
  }

  ~ScopedStdoutCapture() {
    if (capturing_) {
      (void)testing::internal::GetCapturedStdout();
    }
  }

  std::string Stop() {
    capturing_ = false;
    return testing::internal::GetCapturedStdout();
  }

 private:
  bool capturing_;
};

template <typename Run>
void RunAndVerifyAttentionRoute(const char* expected_route, Run&& run) {
  ScopedStdoutCapture capture(true);
  run();
  const std::string debug_output = capture.Stop();
  EXPECT_NE(debug_output.find(expected_route), std::string::npos) << debug_output;
}

bool IsTrtFusedAttentionRouteObservable(int qk_head_size,
                                        int v_head_size,
                                        int sequence_length,
                                        bool has_attention_bias) {
  if (qk_head_size != v_head_size || has_attention_bias) {
    return false;
  }

#if USE_TRT_FUSED_ATTENTION
  if (!HasCudaEnvironment(0)) {
    return false;
  }

  // These tests use the non-flash branch of FusedMHARunnerFP16v2::IsSupported.
  // Keep its SM allowlist, head-size rules, and sequence cap in sync with mha_runner.cu.
  const int sm = GetCudaArchitecture() / 10;
  const bool supported_sm = sm == 70 || sm == 75 || sm == 80 ||
                            sm == 86 || sm == 89;
  return supported_sm &&
         (qk_head_size == 32 || qk_head_size == 64) &&
         !(sm == 70 && qk_head_size == 32) &&
         sequence_length <= 384;
#else
  ORT_UNUSED_PARAMETER(qk_head_size);
  ORT_UNUSED_PARAMETER(v_head_size);
  ORT_UNUSED_PARAMETER(sequence_length);
  ORT_UNUSED_PARAMETER(has_attention_bias);
  return false;
#endif
}

bool IsMemoryEfficientAttentionGeometrySupported(int qk_head_size,
                                                 int v_head_size,
                                                 int sequence_length,
                                                 bool has_attention_bias) {
  return qk_head_size % 8 == 0 &&
         v_head_size % 8 == 0 &&
         qk_head_size <= 1024 &&
         v_head_size <= 1024 &&
         (!has_attention_bias || sequence_length % (4 * sizeof(MLFloat16)) == 0);
}

bool IsMemoryEfficientAttentionRouteObservable(int qk_head_size,
                                               int v_head_size,
                                               int sequence_length,
                                               bool has_attention_bias) {
#if USE_MEMORY_EFFICIENT_ATTENTION
  return HasCudaEnvironment(530) &&
         IsMemoryEfficientAttentionGeometrySupported(
             qk_head_size, v_head_size, sequence_length, has_attention_bias);
#else
  ORT_UNUSED_PARAMETER(qk_head_size);
  ORT_UNUSED_PARAMETER(v_head_size);
  ORT_UNUSED_PARAMETER(sequence_length);
  ORT_UNUSED_PARAMETER(has_attention_bias);
  return false;
#endif
}

bool IsFlashAttentionRouteObservable(int qk_head_size,
                                     int v_head_size,
                                     bool has_attention_bias) {
#if USE_FLASH_ATTENTION
  if (!HasCudaEnvironment(800) ||
      qk_head_size != v_head_size ||
      qk_head_size % 8 != 0 ||
      qk_head_size > 256 ||
      has_attention_bias) {
    return false;
  }

#ifdef ORT_QUICK_BUILD
  return qk_head_size == 128;
#else
  return true;
#endif
#else
  ORT_UNUSED_PARAMETER(qk_head_size);
  ORT_UNUSED_PARAMETER(v_head_size);
  ORT_UNUSED_PARAMETER(has_attention_bias);
  return false;
#endif
}

}  // namespace

#define InvokePackedMultiHeadAttentionTest(use_float16, use_scale) \
  RunPackedMultiHeadAttentionTest(                                 \
      query_data,                                                  \
      key_data,                                                    \
      value_data,                                                  \
      bias_data,                                                   \
      token_offset,                                                \
      cumulative_sequence_length,                                  \
      output_data,                                                 \
      batch_size,                                                  \
      sequence_length,                                             \
      hidden_size,                                                 \
      v_hidden_size,                                               \
      number_of_heads,                                             \
      token_count,                                                 \
      use_float16,                                                 \
      use_scale,                                                   \
      attention_bias_data,                                         \
      broadcast_attention_bias);

static void RunPackedMultiHeadAttentionTest(
    const std::vector<float>& query_data,                    // query:      [token_count, num_heads, 3, head_size]
                                                             //          or [token_count, hidden_size]
    const std::vector<float>& key_data,                      // key:        [token_count, hidden_size]
    const std::vector<float>& value_data,                    // value:      [token_count, v_hidden_size]
    const std::vector<float>& bias_data,                     // bias:       [hidden_size + hidden_size + v_hidden_size]
    const std::vector<int32_t>& token_offset,                // token_offset: [batch_size, sequence_length]
    const std::vector<int32_t>& cumulative_sequence_length,  // cum_seq_len: [batch_size + 1]
    const std::vector<float>& output_data,                   // output:     [token_count, hidden_size]
    int batch_size,
    int sequence_length,
    int hidden_size,
    int v_hidden_size,
    int number_of_heads,
    int token_count,
    bool use_float16,
    bool use_scale,
    const std::vector<float>& attention_bias_data,
    bool broadcast_attention_bias) {
  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);

  int64_t head_size = static_cast<int64_t>(hidden_size / number_of_heads);

  if (enable_cuda) {
    OpTester tester("PackedMultiHeadAttention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));
    if (use_scale) {
      tester.AddAttribute<float>("scale", static_cast<float>(1.f / sqrt(head_size)));
    }

    std::vector<int64_t> packed_qkv_dims = {token_count, number_of_heads, 3, head_size};
    std::vector<int64_t> query_dims = {token_count, hidden_size};
    std::vector<int64_t> key_dims = {token_count, hidden_size};
    std::vector<int64_t> value_dims = {token_count, hidden_size};
    std::vector<int64_t> bias_dims = {hidden_size + hidden_size + v_hidden_size};
    std::vector<int64_t> token_offset_dims = {batch_size, sequence_length};
    std::vector<int64_t> cum_seq_len_dims = {batch_size + 1};
    std::vector<int64_t> attention_bias_data_dims = {batch_size, number_of_heads, sequence_length, sequence_length};
    std::vector<int64_t> broadcast_attention_bias_data_dims = {1, number_of_heads, sequence_length, sequence_length};
    auto& rel_pos_bias_dims = (broadcast_attention_bias ? broadcast_attention_bias_data_dims : attention_bias_data_dims);

    std::vector<int64_t> output_dims = {token_count, v_hidden_size};

    bool is_packed_qkv = (key_data.size() == 0 && value_data.size() == 0);  // packed QKV format

    if (use_float16) {
      if (is_packed_qkv) {
        tester.AddInput<MLFloat16>("query", packed_qkv_dims, ToFloat16(query_data));
        tester.AddOptionalInputEdge<MLFloat16>();
        tester.AddOptionalInputEdge<MLFloat16>();
      } else {
        tester.AddInput<MLFloat16>("query", query_dims, ToFloat16(query_data));
        tester.AddInput<MLFloat16>("key", key_dims, ToFloat16(key_data));
        tester.AddInput<MLFloat16>("value", value_dims, ToFloat16(value_data));
      }

      if (bias_data.size() > 0) {
        tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
      } else {
        tester.AddOptionalInputEdge<MLFloat16>();
      }

      tester.AddInput<int32_t>("token_offset", token_offset_dims, token_offset);
      tester.AddInput<int32_t>("cumulative_sequence_length", cum_seq_len_dims, cumulative_sequence_length);
      if (attention_bias_data.size() > 0) {
        tester.AddInput<MLFloat16>("attention_bias",
                                   rel_pos_bias_dims,
                                   ToFloat16(attention_bias_data));
      }

      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
      tester.SetOutputTolerance(0.005f);
    } else {
      if (is_packed_qkv) {
        tester.AddInput<float>("query", packed_qkv_dims, query_data);
        tester.AddOptionalInputEdge<float>();
        tester.AddOptionalInputEdge<float>();
      } else {
        tester.AddInput<float>("query", query_dims, query_data);
        tester.AddInput<float>("key", key_dims, key_data);
        tester.AddInput<float>("value", value_dims, value_data);
      }

      if (bias_data.size() > 0) {
        tester.AddInput<float>("bias", bias_dims, bias_data);
      } else {
        tester.AddOptionalInputEdge<float>();
      }

      tester.AddInput<int32_t>("token_offset", token_offset_dims, token_offset);
      tester.AddInput<int32_t>("cumulative_sequence_length", cum_seq_len_dims, cumulative_sequence_length);
      if (attention_bias_data.size() > 0) {
        tester.AddInput<float>("attention_bias", rel_pos_bias_dims, attention_bias_data);
      }

      tester.AddOutput<float>("output", output_dims, output_data);
      tester.SetOutputTolerance(0.001f, 0.001f);
    }

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

static void RunPackedMultiHeadAttentionTest(
    const std::vector<float>& query_data,                    // query:      [token_count, num_heads, 3, head_size]
                                                             //          or [token_count, hidden_size]
    const std::vector<float>& key_data,                      // key:        [token_count, hidden_size]
    const std::vector<float>& value_data,                    // value:      [token_count, v_hidden_size]
    const std::vector<float>& bias_data,                     // bias:       [hidden_size + hidden_size + v_hidden_size]
    const std::vector<int32_t>& token_offset,                // token_offset: [batch_size, sequence_length]
    const std::vector<int32_t>& cumulative_sequence_length,  // cum_seq_len: [batch_size + 1]
    const std::vector<float>& output_data,                   // output:     [token_count, hidden_size]
    int batch_size,
    int sequence_length,
    int hidden_size,
    int v_hidden_size,
    int number_of_heads,
    int token_count,
    AttentionKernelType kernel_type,
    const std::vector<float>& attention_bias_data = {},
    bool broadcast_attention_bias = false) {
  const int qk_head_size = hidden_size / number_of_heads;
  const int v_head_size = v_hidden_size / number_of_heads;
  const bool has_attention_bias = !attention_bias_data.empty();

  if (kernel_type == AttentionKernelType::AttentionKernel_TrtFusedAttention) {
    if (!IsTrtFusedAttentionRouteObservable(
            qk_head_size, v_head_size, sequence_length, has_attention_bias)) {
      GTEST_SKIP() << "PackedMultiHeadAttention TRT route is unavailable for this configuration.";
    }

    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableFlashAttention, "1"},
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
            {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"},
            {onnxruntime::contrib::attention::kEnableAttentionKernelDebugInfo, "1"}}};
    RunAndVerifyAttentionRoute("SdpaKernel=TRT_FUSED_ATTENTION", [&]() {
      InvokePackedMultiHeadAttentionTest(true, true);
    });
    RunAndVerifyAttentionRoute("SdpaKernel=TRT_FUSED_ATTENTION", [&]() {
      InvokePackedMultiHeadAttentionTest(true, false);
    });
  }

  if (kernel_type == AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention) {
    const bool geometry_supports_mea = IsMemoryEfficientAttentionGeometrySupported(
        qk_head_size, v_head_size, sequence_length, has_attention_bias);
    if (geometry_supports_mea &&
        !IsMemoryEfficientAttentionRouteObservable(
            qk_head_size, v_head_size, sequence_length, has_attention_bias)) {
      GTEST_SKIP() << "PackedMultiHeadAttention MEA route is unavailable in this build or CUDA environment.";
    }

    if (!geometry_supports_mea && !HasCudaEnvironment(530)) {
      GTEST_SKIP() << "PackedMultiHeadAttention invalid-geometry fallback requires a CUDA device.";
    }

    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableFlashAttention, "1"},
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
            {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "0"},
            {onnxruntime::contrib::attention::kEnableAttentionKernelDebugInfo, "1"}}};
    const char* expected_route =
        geometry_supports_mea ? "SdpaKernel=EFFICIENT_ATTENTION" : "SdpaKernel=MATH";
    RunAndVerifyAttentionRoute(expected_route, [&]() {
      InvokePackedMultiHeadAttentionTest(true, true);
    });
    RunAndVerifyAttentionRoute(expected_route, [&]() {
      InvokePackedMultiHeadAttentionTest(true, false);
    });
  }

  if (kernel_type == AttentionKernelType::AttentionKernel_FlashAttention) {
    if (!IsFlashAttentionRouteObservable(qk_head_size, v_head_size, has_attention_bias)) {
      GTEST_SKIP() << "PackedMultiHeadAttention Flash route is unavailable in this build or CUDA environment.";
    }

    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableFlashAttention, "0"},
            {onnxruntime::contrib::attention::kMinSeqLenForFlashAttentionPackedQKV, "0"},
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
            {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "0"},
            {onnxruntime::contrib::attention::kEnableAttentionKernelDebugInfo, "1"}}};
    RunAndVerifyAttentionRoute("SdpaKernel=FLASH_ATTENTION", [&]() {
      InvokePackedMultiHeadAttentionTest(true, true);
    });
  }

  if (kernel_type == AttentionKernelType::AttentionKernel_Unfused) {
    if (!HasCudaEnvironment(530)) {
      GTEST_SKIP() << "PackedMultiHeadAttention MATH route requires a CUDA device with FP16 support.";
    }

    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableFlashAttention, "1"},
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
            {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"},
            {onnxruntime::contrib::attention::kEnableAttentionKernelDebugInfo, "1"}}};
    RunAndVerifyAttentionRoute("SdpaKernel=MATH", [&]() {
      InvokePackedMultiHeadAttentionTest(true, true);
    });
    RunAndVerifyAttentionRoute("SdpaKernel=MATH", [&]() {
      InvokePackedMultiHeadAttentionTest(false, false);
    });
  }

  if (kernel_type == AttentionKernelType::AttentionKernel_Default) {
    InvokePackedMultiHeadAttentionTest(true, false);
    InvokePackedMultiHeadAttentionTest(false, true);
  }
}

TEST(PackedMultiHeadAttentionTest, EmptyTokensAndSequence_CUDA) {
  if (!HasCudaEnvironment(0)) {
    GTEST_SKIP() << "PackedMultiHeadAttention empty-output test requires a CUDA device.";
  }

  constexpr int kBatchSize = 2;
  constexpr int kNumHeads = 2;
  constexpr int kHeadSize = 16;
  constexpr int kHiddenSize = kNumHeads * kHeadSize;

  OpTester tester("PackedMultiHeadAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", kNumHeads);
  tester.AddInput<float>("query", {0, kNumHeads, 3, kHeadSize}, {});
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<float>();
  tester.AddInput<int32_t>("token_offset", {kBatchSize, 0}, {});
  tester.AddInput<int32_t>("cumulative_sequence_length", {kBatchSize + 1}, {0, 0, 0});
  tester.AddOutput<float>("output", {0, kHiddenSize}, {});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(PackedMultiHeadAttentionTest, InvalidTokenOffset_Unfused_CUDA) {
  if (!HasCudaEnvironment(0)) {
    GTEST_SKIP() << "PackedMultiHeadAttention token offset validation requires a CUDA device.";
  }

  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::attention::kDisableFlashAttention, "1"},
          {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "1"},
          {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "1"},
          {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
          {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"}}};

  constexpr int kTokenCount = 1;
  constexpr int kBatchSize = 1;
  constexpr int kSequenceLength = 2;
  constexpr int kNumHeads = 1;
  constexpr int kHeadSize = 8;

  for (const bool use_packed_qkv : {false, true}) {
    for (const std::vector<int32_t>& token_offset : {
             std::vector<int32_t>{-1, 1},
             std::vector<int32_t>{0, kBatchSize * kSequenceLength}}) {
      SCOPED_TRACE(use_packed_qkv ? "packed QKV" : "separate Q/K/V");
      SCOPED_TRACE(token_offset[0]);
      SCOPED_TRACE(token_offset[1]);

      OpTester tester("PackedMultiHeadAttention", 1, onnxruntime::kMSDomain);
      tester.AddAttribute<int64_t>("num_heads", kNumHeads);
      if (use_packed_qkv) {
        tester.AddInput<float>("query", {kTokenCount, kNumHeads, 3, kHeadSize},
                               std::vector<float>(kTokenCount * kNumHeads * 3 * kHeadSize));
        tester.AddOptionalInputEdge<float>();
        tester.AddOptionalInputEdge<float>();
      } else {
        tester.AddInput<float>("query", {kTokenCount, kHeadSize}, std::vector<float>(kHeadSize));
        tester.AddInput<float>("key", {kTokenCount, kHeadSize}, std::vector<float>(kHeadSize));
        tester.AddInput<float>("value", {kTokenCount, kHeadSize}, std::vector<float>(kHeadSize));
      }
      tester.AddOptionalInputEdge<float>();
      tester.AddInput<int32_t>("token_offset", {kBatchSize, kSequenceLength}, token_offset);
      tester.AddInput<int32_t>("cumulative_sequence_length", {kBatchSize + 1}, {0, kTokenCount});
      tester.AddOutput<float>("output", {kTokenCount, kHeadSize}, std::vector<float>(kHeadSize));

      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(
          OpTester::ExpectResult::kExpectFailure,
          "PackedMultiHeadAttention token_offset values must be in [0, B * S).",
          {}, nullptr, &execution_providers);
    }
  }
}

TEST(PackedMultiHeadAttentionTest, TokenOffsetValidationIndexProgressionUsesInt64) {
  constexpr int64_t index = 2147221504;
  constexpr int64_t next_index =
      AdvanceTokenOffsetValidationIndex(index, 1024, 256);

  EXPECT_EQ(next_index, 2147483648LL);
  EXPECT_GT(next_index, std::numeric_limits<int32_t>::max());
}

TEST(PackedMultiHeadAttentionTest, ValidTokenOffset_UnfusedCudaGraphCaptureAndReplay) {
  if (!HasCudaEnvironment(0)) {
    GTEST_SKIP() << "PackedMultiHeadAttention CUDA graph test requires a CUDA device.";
  }

  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::attention::kDisableFlashAttention, "1"},
          {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "1"},
          {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "1"},
          {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
          {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"}}};

  constexpr int64_t kTokenCount = 2;
  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kSequenceLength = 2;
  constexpr int64_t kNumHeads = 1;
  constexpr int64_t kHeadSize = 8;

  auto make_tensor_type = [](int32_t element_type, std::initializer_list<int64_t> shape) {
    ONNX_NAMESPACE::TypeProto type;
    auto* tensor_type = type.mutable_tensor_type();
    tensor_type->set_elem_type(element_type);
    for (int64_t dimension : shape) {
      tensor_type->mutable_shape()->add_dim()->set_dim_value(dimension);
    }
    return type;
  };

  auto query_type = make_tensor_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {kTokenCount, kNumHeads, 3, kHeadSize});
  auto token_offset_type = make_tensor_type(
      ONNX_NAMESPACE::TensorProto_DataType_INT32, {kBatchSize, kSequenceLength});
  auto cumulative_sequence_length_type = make_tensor_type(
      ONNX_NAMESPACE::TensorProto_DataType_INT32, {kBatchSize + 1});
  auto output_type = make_tensor_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {kTokenCount, kNumHeads * kHeadSize});

  Model model("packed_multihead_attention_cuda_graph", true, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 17}, {kMSDomain, 1}},
              {}, DefaultLoggingManager().DefaultLogger(), ModelOptions(true, true));
  auto& graph = model.MainGraph();
  auto& empty = graph.GetOrCreateNodeArg("", nullptr);
  std::vector<NodeArg*> inputs{
      &graph.GetOrCreateNodeArg("query", &query_type),
      &empty,
      &empty,
      &empty,
      &graph.GetOrCreateNodeArg("token_offset", &token_offset_type),
      &graph.GetOrCreateNodeArg("cumulative_sequence_length", &cumulative_sequence_length_type),
      &empty};
  std::vector<NodeArg*> outputs{
      &graph.GetOrCreateNodeArg("output", &output_type)};
  auto& node = graph.AddNode(
      "packed_multihead_attention", "PackedMultiHeadAttention", "",
      inputs, outputs, nullptr, kMSDomain);
  node.AddAttribute("num_heads", kNumHeads);
  ASSERT_STATUS_OK(graph.Resolve());

  std::string model_data;
  ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));

  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.do_copy_in_default_stream = true;
  provider_options.enable_cuda_graph = true;
  auto cuda_ep = CudaExecutionProviderWithOptions(&provider_options);
  ASSERT_NE(cuda_ep, nullptr);
  IExecutionProvider* cuda_ep_ptr = cuda_ep.get();

  SessionOptions session_options;
  InferenceSession session(session_options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(cuda_ep)));
  std::istringstream model_stream(model_data);
  ASSERT_STATUS_OK(session.Load(model_stream));
  ASSERT_STATUS_OK(session.Initialize());

  auto gpu_allocators = cuda_ep_ptr->CreatePreferredAllocators();
  auto gpu_allocator = std::find_if(gpu_allocators.begin(), gpu_allocators.end(), [](const auto& allocator) {
    return allocator->Info().device.Type() == OrtDevice::GPU &&
           allocator->Info().alloc_type != OrtAllocatorType::OrtReadOnlyAllocator &&
           allocator->Info().mem_type == OrtMemTypeDefault;
  });
  ASSERT_NE(gpu_allocator, gpu_allocators.end());
  auto allocator = session.GetAllocator((*gpu_allocator)->Info());
  ASSERT_NE(allocator, nullptr);
  auto cpu_allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];

  auto make_gpu_value = [&](const auto& values, const TensorShape& shape) {
    using Element = typename std::decay_t<decltype(values)>::value_type;
    Tensor cpu_tensor(
        DataTypeImpl::GetType<Element>(), shape,
        const_cast<Element*>(values.data()), cpu_allocator->Info());
    Tensor gpu_tensor(DataTypeImpl::GetType<Element>(), shape, allocator);
    ORT_THROW_IF_ERROR(cuda_ep_ptr->GetDataTransfer()->CopyTensor(cpu_tensor, gpu_tensor));
    OrtValue result;
    Tensor::InitOrtValue(std::move(gpu_tensor), result);
    return result;
  };

  auto query = make_gpu_value(
      std::vector<float>(kTokenCount * kNumHeads * 3 * kHeadSize),
      TensorShape{kTokenCount, kNumHeads, 3, kHeadSize});
  auto token_offset = make_gpu_value(
      std::vector<int32_t>{0, 1}, TensorShape{kBatchSize, kSequenceLength});
  auto cumulative_sequence_length = make_gpu_value(
      std::vector<int32_t>{0, 2}, TensorShape{kBatchSize + 1});
  Tensor output_tensor(
      DataTypeImpl::GetType<float>(),
      TensorShape{kTokenCount, kNumHeads * kHeadSize}, allocator);
  OrtValue output;
  Tensor::InitOrtValue(std::move(output_tensor), output);

  std::unique_ptr<IOBinding> binding;
  ASSERT_STATUS_OK(session.NewIOBinding(&binding));
  ASSERT_STATUS_OK(binding->BindInput("query", query));
  ASSERT_STATUS_OK(binding->BindInput("token_offset", token_offset));
  ASSERT_STATUS_OK(binding->BindInput("cumulative_sequence_length", cumulative_sequence_length));
  ASSERT_STATUS_OK(binding->BindOutput("output", output));
  ASSERT_STATUS_OK(binding->SynchronizeInputs());

  RunOptions run_options;
  ASSERT_STATUS_OK(run_options.config_options.AddConfigEntry("gpu_graph_id", "1"));
  for (int i = 0; i < 4; ++i) {
    ASSERT_STATUS_OK(session.Run(run_options, *binding));
  }
  EXPECT_TRUE(cuda_ep_ptr->IsGraphCaptured(1));
}

TEST(PackedMultiHeadAttentionTest, PackedQKV_NoPadding_NoBias_trt) {
  AttentionTestData data;
  GetSelfAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedQKV(data);
  std::vector<float> empty_data = {};
  std::vector<int32_t> token_offset{0, 1, 2, 3};
  std::vector<int32_t> cum_seq_len{0, 2, 4};

  RunPackedMultiHeadAttentionTest(
      data.qkv_data,
      empty_data,
      empty_data,
      empty_data,
      token_offset,
      cum_seq_len,
      data.fp16_output_data,
      data.batch_size,
      data.sequence_length,
      data.hidden_size,
      data.v_hidden_size,
      data.num_heads,
      data.batch_size * data.sequence_length,
      AttentionKernelType::AttentionKernel_TrtFusedAttention);
}

TEST(PackedMultiHeadAttentionTest, PackedQKV_NoPadding_NoBias_cutlass) {
  AttentionTestData data;
  GetSelfAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedQKV(data);
  std::vector<float> empty_data = {};
  std::vector<int32_t> token_offset{0, 1, 2, 3};
  std::vector<int32_t> cum_seq_len{0, 2, 4};

  RunPackedMultiHeadAttentionTest(
      data.qkv_data,
      empty_data,
      empty_data,
      empty_data,
      token_offset,
      cum_seq_len,
      data.fp16_output_data,
      data.batch_size,
      data.sequence_length,
      data.hidden_size,
      data.v_hidden_size,
      data.num_heads,
      data.batch_size * data.sequence_length,
      AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention);
}

TEST(PackedMultiHeadAttentionTest, PackedQKV_NoPadding_NoBias_unfused) {
  AttentionTestData data;
  GetSelfAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedQKV(data);
  std::vector<float> empty_data = {};
  std::vector<int32_t> token_offset{0, 1, 2, 3};
  std::vector<int32_t> cum_seq_len{0, 2, 4};

  RunPackedMultiHeadAttentionTest(
      data.qkv_data,
      empty_data,
      empty_data,
      empty_data,
      token_offset,
      cum_seq_len,
      data.fp16_output_data,
      data.batch_size,
      data.sequence_length,
      data.hidden_size,
      data.v_hidden_size,
      data.num_heads,
      data.batch_size * data.sequence_length,
      AttentionKernelType::AttentionKernel_Unfused);
}

TEST(PackedMultiHeadAttentionTest, Q_K_V_NoPadding_NoBias_trt) {
  AttentionTestData data;
  GetSelfAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedQKV(data);
  std::vector<float> empty_data = {};
  std::vector<int32_t> token_offset{0, 1, 2, 3};
  std::vector<int32_t> cum_seq_len{0, 2, 4};

  RunPackedMultiHeadAttentionTest(
      data.query_data,
      data.key_data,
      data.value_data,
      empty_data,
      token_offset,
      cum_seq_len,
      data.fp16_output_data,
      data.batch_size,
      data.sequence_length,
      data.hidden_size,
      data.v_hidden_size,
      data.num_heads,
      data.batch_size * data.sequence_length,
      AttentionKernelType::AttentionKernel_TrtFusedAttention);
}

TEST(PackedMultiHeadAttentionTest, Q_K_V_NoPadding_Bias_AttnBias_InvalidHeadFallback) {
  AttentionTestData data;
  GetAttentionDataCutlassAttnBias(data);
  std::vector<int32_t> token_offset{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int32_t> cum_seq_len{0, 8};

  RunPackedMultiHeadAttentionTest(
      data.query_data,
      data.key_data,
      data.value_data,
      data.bias_data,
      token_offset,
      cum_seq_len,
      data.fp16_output_data,
      data.batch_size,
      data.sequence_length,
      data.hidden_size,
      data.v_hidden_size,
      data.num_heads,
      data.batch_size * data.sequence_length,
      AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention,
      data.attention_bias_data,
      data.broadcast_attention_bias);
}

TEST(PackedMultiHeadAttentionTest, Q_K_V_NoPadding_Bias_AttnBias_unfused) {
  AttentionTestData data;
  GetAttentionDataCutlassAttnBias(data);
  std::vector<int32_t> token_offset{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int32_t> cum_seq_len{0, 8};

  RunPackedMultiHeadAttentionTest(
      data.query_data,
      data.key_data,
      data.value_data,
      data.bias_data,
      token_offset,
      cum_seq_len,
      data.fp16_output_data,
      data.batch_size,
      data.sequence_length,
      data.hidden_size,
      data.v_hidden_size,
      data.num_heads,
      data.batch_size * data.sequence_length,
      AttentionKernelType::AttentionKernel_Unfused,
      data.attention_bias_data,
      data.broadcast_attention_bias);
}

TEST(PackedMultiHeadAttentionTest, PackedQKV_Padding_NoBias_trt) {
  PackedAttentionTestData data;
  GetPackedMultiHeadAttentionData_Batch2_HeadSize32_NoAttnBias(data);
  std::vector<float> empty_data = {};

  RunPackedMultiHeadAttentionTest(
      data.qkv_data,
      empty_data,
      empty_data,
      empty_data,
      data.token_offset,
      data.cumulative_sequence_length,
      data.fp16_output_data,
      data.batch_size,
      data.sequence_length,
      data.hidden_size,
      data.v_hidden_size,
      data.num_heads,
      data.token_count,
      AttentionKernelType::AttentionKernel_TrtFusedAttention);
}

TEST(PackedMultiHeadAttentionTest, PackedQKV_Padding_NoBias_cutlass) {
  PackedAttentionTestData data;
  GetPackedMultiHeadAttentionData_Batch2_HeadSize32_NoAttnBias(data);
  std::vector<float> empty_data = {};

  RunPackedMultiHeadAttentionTest(
      data.qkv_data,
      empty_data,
      empty_data,
      empty_data,
      data.token_offset,
      data.cumulative_sequence_length,
      data.fp16_output_data,
      data.batch_size,
      data.sequence_length,
      data.hidden_size,
      data.v_hidden_size,
      data.num_heads,
      data.token_count,
      AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention);
}

TEST(PackedMultiHeadAttentionTest, PackedQKV_Padding_NoBias_FlashAttention) {
  PackedAttentionTestData data;
  GetPackedMultiHeadAttentionData_Batch2_HeadSize32_NoAttnBias(data);
  std::vector<float> empty_data = {};

  RunPackedMultiHeadAttentionTest(
      data.qkv_data,
      empty_data,
      empty_data,
      empty_data,
      data.token_offset,
      data.cumulative_sequence_length,
      data.fp16_output_data,
      data.batch_size,
      data.sequence_length,
      data.hidden_size,
      data.v_hidden_size,
      data.num_heads,
      data.token_count,
      AttentionKernelType::AttentionKernel_FlashAttention);
}

TEST(PackedMultiHeadAttentionTest, PackedQKV_Padding_NoBias_unfused) {
  PackedAttentionTestData data;
  GetPackedMultiHeadAttentionData_Batch2_HeadSize32_NoAttnBias(data);
  std::vector<float> empty_data = {};

  RunPackedMultiHeadAttentionTest(
      data.qkv_data,
      empty_data,
      empty_data,
      empty_data,
      data.token_offset,
      data.cumulative_sequence_length,
      data.fp16_output_data,
      data.batch_size,
      data.sequence_length,
      data.hidden_size,
      data.v_hidden_size,
      data.num_heads,
      data.token_count,
      AttentionKernelType::AttentionKernel_Unfused);
}

TEST(PackedMultiHeadAttentionTest, PackedQKV_Padding_NoBias_AttnBias) {
  PackedAttentionTestData data;
  GetPackedMultiHeadAttentionData_Batch2_HeadSize8_AttnBias(data);
  std::vector<float> empty_data = {};

  RunPackedMultiHeadAttentionTest(
      data.qkv_data,
      empty_data,
      empty_data,
      empty_data,
      data.token_offset,
      data.cumulative_sequence_length,
      data.fp16_output_data,
      data.batch_size,
      data.sequence_length,
      data.hidden_size,
      data.v_hidden_size,
      data.num_heads,
      data.token_count,
      AttentionKernelType::AttentionKernel_Default,
      data.attention_bias_data,
      data.broadcast_attention_bias);
}

TEST(PackedMultiHeadAttentionTest, PackedQKV_Padding_NoBias_BroadcastAttnBias_cutlass) {
  PackedAttentionTestData data;
  GetPackedMultiHeadAttentionData_Batch2_HeadSize8_BroadcastAttnBias(data);
  std::vector<float> empty_data = {};

  RunPackedMultiHeadAttentionTest(
      data.qkv_data,
      empty_data,
      empty_data,
      empty_data,
      data.token_offset,
      data.cumulative_sequence_length,
      data.fp16_output_data,
      data.batch_size,
      data.sequence_length,
      data.hidden_size,
      data.v_hidden_size,
      data.num_heads,
      data.token_count,
      AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention,
      data.attention_bias_data,
      data.broadcast_attention_bias);
}

TEST(PackedMultiHeadAttentionTest, PackedQKV_Padding_NoBias_BroadcastAttnBias_unfused) {
  PackedAttentionTestData data;
  GetPackedMultiHeadAttentionData_Batch2_HeadSize8_BroadcastAttnBias(data);
  std::vector<float> empty_data = {};

  RunPackedMultiHeadAttentionTest(
      data.qkv_data,
      empty_data,
      empty_data,
      empty_data,
      data.token_offset,
      data.cumulative_sequence_length,
      data.fp16_output_data,
      data.batch_size,
      data.sequence_length,
      data.hidden_size,
      data.v_hidden_size,
      data.num_heads,
      data.token_count,
      AttentionKernelType::AttentionKernel_Unfused,
      data.attention_bias_data,
      data.broadcast_attention_bias);
}

}  // namespace test
}  // namespace onnxruntime
