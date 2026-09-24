// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_CONTRIB_OPS) && !defined(USE_CUDA_MINIMAL) && !defined(BUILD_CUDA_EP_AS_PLUGIN)

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "core/framework/allocator.h"
#include "core/framework/session_state.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime {
namespace test {
namespace {

TEST(GroupQueryAttentionWorkspace, OutOfBoundFirstRunStillCachesDeclaredSlot) {
  ScopedEnvironmentVariables env_vars{{{"ORT_ENABLE_XQA", "0"}}};
  OrtCUDAProviderOptionsV2 cuda_options{};
  cuda_options.sdpa_kernel = static_cast<int>(contrib::attention::AttentionBackend::MATH);
  cuda_options.do_copy_in_default_stream = true;
  auto cuda_ep = CudaExecutionProviderWithOptions(&cuda_options);
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA execution provider is unavailable.";
  }

  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("com.microsoft");
  opset->set_version(1);
  auto* graph = model.mutable_graph();
  graph->set_name("gqa_workspace_scalar_bound");
  auto* node = graph->add_node();
  node->set_name("gqa");
  node->set_op_type("GroupQueryAttention");
  node->set_domain("com.microsoft");
  for (const auto& [name, value] : {std::pair{"num_heads", 4}, std::pair{"kv_num_heads", 2}}) {
    auto* attribute = node->add_attribute();
    attribute->set_name(name);
    attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attribute->set_i(value);
  }
  const std::array<std::string, 7> input_names{
      "query", "key", "value", "past_key", "past_value", "seqlens_k", "total_sequence_length"};
  const std::array<TensorShapeVector, 7> input_shapes{
      TensorShapeVector{1, 4, 256}, TensorShapeVector{1, 4, 128}, TensorShapeVector{1, 4, 128},
      TensorShapeVector{1, 2, 256, 64}, TensorShapeVector{1, 2, 256, 64},
      TensorShapeVector{1}, TensorShapeVector{}};
  for (size_t i = 0; i < input_names.size(); ++i) {
    node->add_input(input_names[i]);
    auto* input = graph->add_input();
    input->set_name(input_names[i]);
    auto* type = input->mutable_type()->mutable_tensor_type();
    type->set_elem_type(i < 5 ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT16
                              : ONNX_NAMESPACE::TensorProto_DataType_INT32);
    auto* shape = type->mutable_shape();
    for (const auto dim : input_shapes[i]) {
      shape->add_dim()->set_dim_value(dim);
    }
  }
  const std::vector<std::string> output_names{"output", "present_key", "present_value"};
  for (const auto& name : output_names) {
    node->add_output(name);
    auto* output = graph->add_output();
    output->set_name(name);
    output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  }
  const auto model_bytes = model.SerializeAsString();

  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsEnableStaticWorkspacePreallocation, "1"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsCudaGqaWorkspaceMaxTotalSequenceLength, "64"));
  InferenceSessionWrapper session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(cuda_ep)));
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  const auto& state = session.GetSessionState();
  ASSERT_TRUE(state.GetEnableMemoryPattern());
  const Node* gqa_node = nullptr;
  for (const auto& candidate : session.GetGraph().Nodes()) {
    if (candidate.OpType() == "GroupQueryAttention") {
      gqa_node = &candidate;
    }
  }
  ASSERT_NE(gqa_node, nullptr);
  ASSERT_EQ(gqa_node->GetExecutionProviderType(), kCudaExecutionProvider);
  const auto& plans = state.GetExecutionPlan()->workspace_allocation_plan.at(gqa_node->Index());
  ASSERT_EQ(plans.size(), 1u);
  const auto& plan = plans.front();
  ASSERT_GT(plan.size_bytes, 0u);

  auto allocator = std::make_shared<CPUAllocator>();
  InlinedVector<OrtValue> ordered_feeds(input_names.size());
  InlinedVector<int> feed_indices(input_names.size());
  for (size_t i = 0; i < input_names.size(); ++i) {
    ASSERT_STATUS_OK(state.GetOrtValueNameIdxMap().GetIdx(input_names[i], feed_indices[i]));
    if (i < 5) {
      std::vector<MLFloat16> data(static_cast<size_t>(TensorShape(input_shapes[i]).Size()),
                                  MLFloat16(i == 2 ? 1.0f : 0.0f));
      CreateMLValue<MLFloat16>(allocator, input_shapes[i], data, &ordered_feeds[i]);
    }
  }

  const MemoryPatternGroup* cached_patterns = nullptr;
  for (const int32_t total_length : {128, 64, 128, 64}) {
    SCOPED_TRACE(total_length);
    CreateMLValue<int32_t>(allocator, input_shapes[5], {total_length - 1}, &ordered_feeds[5]);
    CreateMLValue<int32_t>(allocator, input_shapes[6], {total_length}, &ordered_feeds[6]);
    std::vector<OrtValue> outputs;
    ASSERT_STATUS_OK(session.Run(RunOptions{}, input_names, ordered_feeds, output_names, &outputs));
    ASSERT_EQ(outputs.size(), output_names.size());
    const auto& output = outputs[0].Get<Tensor>();
    ASSERT_EQ(output.Shape(), TensorShape({1, 4, 256}));
    // Zero Q/K gives uniform causal attention; only the four new V rows are one.
    const auto values = output.DataAsSpan<MLFloat16>();
    for (size_t row = 0; row < 4; ++row) {
      const float expected =
          static_cast<float>(row + 1) / (static_cast<float>(total_length - 3) + static_cast<float>(row));
      for (size_t col = 0; col < 256; ++col) {
        EXPECT_NEAR(values[row * 256 + col].ToFloat(), expected, 0.001f);
      }
    }

    const InlinedHashMap<int, TensorShape>* inferred_shapes = nullptr;
    const auto* patterns = state.GetMemoryPatternGroup(ordered_feeds, feed_indices, inferred_shapes);
    ASSERT_NE(patterns, nullptr);
    if (cached_patterns != nullptr) {
      EXPECT_EQ(patterns, cached_patterns);
    }
    cached_patterns = patterns;
    const auto* device_pattern = patterns->GetPatterns(plan.location);
    ASSERT_NE(device_pattern, nullptr);
    const auto* block = device_pattern->GetBlock(plan.pattern_id);
    ASSERT_NE(block, nullptr);
    EXPECT_EQ(block->size_, plan.allocation_bytes);
  }
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif
