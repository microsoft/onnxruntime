// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/graph/model.h"
#include "core/graph/node_attr_utils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "core/session/IOBinding.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime {
namespace test {

namespace {

void RunSparseAttentionInvalidInputTest(const std::vector<int32_t>& total_key_lengths_data,
                                        const std::vector<int64_t>& total_key_lengths_dims,
                                        const std::string& expected_error,
                                        int32_t total_sequence_length = 4) {
  OpTester test("SparseAttention", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("num_heads", 2);
  test.AddAttribute<int64_t>("kv_num_heads", 2);
  test.AddAttribute<int64_t>("sparse_block_size", 1);
  test.AddAttribute<float>("scale", 1.0f);
  test.AddAttribute<int64_t>("do_rotary", 0);
  test.AddAttribute<int64_t>("rotary_interleaved", 0);

  test.AddInput<float>("query", {1, 1, 16}, std::vector<float>(16, 0.0f));
  test.AddInput<float>("key", {1, 1, 16}, std::vector<float>(16, 0.0f));
  test.AddInput<float>("value", {1, 1, 16}, std::vector<float>(16, 0.0f));
  test.AddInput<float>("past_key", {1, 2, 4, 8}, std::vector<float>(64, 0.0f));
  test.AddInput<float>("past_value", {1, 2, 4, 8}, std::vector<float>(64, 0.0f));
  test.AddInput<int32_t>("block_row_indices", {1, 5}, {0, 1, 2, 3, 4});
  test.AddInput<int32_t>("block_col_indices", {1, 1}, {0});
  test.AddInput<int32_t>("total_sequence_length", {1}, {total_sequence_length});
  test.AddInput<int32_t>("key_total_sequence_lengths", total_key_lengths_dims, total_key_lengths_data);
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();

  test.AddOutput<float>("output", {1, 1, 16}, std::vector<float>(16, 0.0f));
  test.AddOutput<float>("present_key", {1, 2, 4, 8}, std::vector<float>(64, 0.0f));
  test.AddOutput<float>("present_value", {1, 2, 4, 8}, std::vector<float>(64, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, expected_error, {}, nullptr, &execution_providers);
}

void RunSparseAttentionPromptInputTest(const std::vector<int32_t>& total_key_lengths_data,
                                       int64_t batch_size,
                                       int64_t sequence_length,
                                       int32_t total_sequence_length) {
  std::unordered_map<std::string, int> domain_to_version = {{onnxruntime::kOnnxDomain, 13},
                                                            {onnxruntime::kMSDomain, 1}};
  std::vector<ONNX_NAMESPACE::FunctionProto> model_specific_functions;
  auto model = std::make_unique<Model>("sparse_attention_shared_buffer_test", true, ModelMetaData(), PathString(),
                                       IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
                                       model_specific_functions, DefaultLoggingManager().DefaultLogger(),
                                       ModelOptions(true, true));
  onnxruntime::Graph& graph = model->MainGraph();

  ONNX_NAMESPACE::TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  ONNX_NAMESPACE::TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);

  auto& query_arg = graph.GetOrCreateNodeArg("query", &tensor_float);
  auto& key_arg = graph.GetOrCreateNodeArg("key", &tensor_float);
  auto& value_arg = graph.GetOrCreateNodeArg("value", &tensor_float);
  auto& past_key_arg = graph.GetOrCreateNodeArg("past_key", &tensor_float);
  auto& past_value_arg = graph.GetOrCreateNodeArg("past_value", &tensor_float);
  auto& block_row_indices_arg = graph.GetOrCreateNodeArg("block_row_indices", &tensor_int32);
  auto& block_col_indices_arg = graph.GetOrCreateNodeArg("block_col_indices", &tensor_int32);
  auto& total_sequence_length_arg = graph.GetOrCreateNodeArg("total_sequence_length", &tensor_int32);
  auto& key_total_sequence_lengths_arg = graph.GetOrCreateNodeArg("key_total_sequence_lengths", &tensor_int32);
  auto& cos_cache_arg = graph.GetOrCreateNodeArg("cos_cache", &tensor_float);
  auto& sin_cache_arg = graph.GetOrCreateNodeArg("sin_cache", &tensor_float);
  std::vector<onnxruntime::NodeArg*> input_defs = {&query_arg,
                                                   &key_arg,
                                                   &value_arg,
                                                   &past_key_arg,
                                                   &past_value_arg,
                                                   &block_row_indices_arg,
                                                   &block_col_indices_arg,
                                                   &total_sequence_length_arg,
                                                   &key_total_sequence_lengths_arg,
                                                   &cos_cache_arg,
                                                   &sin_cache_arg};

  auto& output_arg = graph.GetOrCreateNodeArg("output", &tensor_float);
  auto& present_key_arg = graph.GetOrCreateNodeArg("present_key", &tensor_float);
  auto& present_value_arg = graph.GetOrCreateNodeArg("present_value", &tensor_float);
  std::vector<onnxruntime::NodeArg*> output_defs = {&output_arg, &present_key_arg, &present_value_arg};

  NodeAttributes attrs{
      {"num_heads", utils::MakeAttribute("num_heads", int64_t{2})},
      {"kv_num_heads", utils::MakeAttribute("kv_num_heads", int64_t{2})},
      {"sparse_block_size", utils::MakeAttribute("sparse_block_size", int64_t{1})},
      {"scale", utils::MakeAttribute("scale", 1.0f)},
      {"do_rotary", utils::MakeAttribute("do_rotary", int64_t{0})},
      {"rotary_interleaved", utils::MakeAttribute("rotary_interleaved", int64_t{0})},
  };

  auto& node = graph.AddNode("node1", "SparseAttention", "SparseAttention shared-buffer test",
                             input_defs, output_defs, &attrs, onnxruntime::kMSDomain);
  node.SetExecutionProviderType(kCpuExecutionProvider);
  ASSERT_STATUS_OK(graph.Resolve());

  std::string model_string;
  model->ToProto().SerializeToString(&model_string);
  std::stringstream model_stream(model_string);

  SessionOptions session_options;
  session_options.session_logid = "SparseAttentionSharedBufferTest";
  InferenceSession session(session_options, GetEnvironment());
  ASSERT_STATUS_OK(session.Load(model_stream));
  ASSERT_STATUS_OK(session.Initialize());

  const int64_t hidden_size = 16;
  const int64_t kv_num_heads = 2;
  const int64_t head_size = hidden_size / kv_num_heads;
  const int64_t max_cache_sequence_length = total_sequence_length;

  const std::vector<int64_t> qkv_dims = {batch_size, sequence_length, hidden_size};
  const std::vector<int64_t> cache_dims = {batch_size, kv_num_heads, max_cache_sequence_length, head_size};
  const std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};
  const std::vector<int64_t> block_row_dims = {1, 6};
  const std::vector<int64_t> block_col_dims = {1, 15};
  const std::vector<int64_t> scalar_dims = {1};
  const std::vector<int64_t> total_key_lengths_dims = {batch_size};
  const std::vector<int64_t> rotary_cache_dims = {1, head_size / 2};

  auto cpu_alloc = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];

  std::vector<float> query_data(static_cast<size_t>(batch_size * sequence_length * hidden_size), 0.0f);
  std::vector<float> key_data(static_cast<size_t>(batch_size * sequence_length * hidden_size), 0.0f);
  std::vector<float> value_data(static_cast<size_t>(batch_size * sequence_length * hidden_size), 0.0f);
  std::vector<float> past_key_data(static_cast<size_t>(batch_size * kv_num_heads * max_cache_sequence_length * head_size), 0.0f);
  std::vector<float> past_value_data(static_cast<size_t>(batch_size * kv_num_heads * max_cache_sequence_length * head_size), 0.0f);
  std::vector<float> output_data(static_cast<size_t>(batch_size * sequence_length * hidden_size), 0.0f);
  std::vector<float> rotary_cache_data(static_cast<size_t>(head_size / 2), 0.0f);

  OrtValue query_value;
  CreateMLValue<float>(cpu_alloc, qkv_dims, query_data, &query_value);
  OrtValue key_value;
  CreateMLValue<float>(cpu_alloc, qkv_dims, key_data, &key_value);
  OrtValue value_value;
  CreateMLValue<float>(cpu_alloc, qkv_dims, value_data, &value_value);
  OrtValue past_key_value;
  CreateMLValue<float>(cache_dims, past_key_data.data(), cpu_alloc->Info(), &past_key_value);
  OrtValue past_value_value;
  CreateMLValue<float>(cache_dims, past_value_data.data(), cpu_alloc->Info(), &past_value_value);
  OrtValue block_row_indices_value;
  CreateMLValue<int32_t>(cpu_alloc, block_row_dims, std::vector<int32_t>{0, 1, 3, 6, 10, 15}, &block_row_indices_value);
  OrtValue block_col_indices_value;
  CreateMLValue<int32_t>(cpu_alloc, block_col_dims, std::vector<int32_t>{0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4}, &block_col_indices_value);
  OrtValue total_sequence_length_value;
  CreateMLValue<int32_t>(cpu_alloc, scalar_dims, std::vector<int32_t>{total_sequence_length}, &total_sequence_length_value);
  OrtValue total_key_lengths_value;
  CreateMLValue<int32_t>(cpu_alloc, total_key_lengths_dims, total_key_lengths_data, &total_key_lengths_value);
  OrtValue cos_cache_value;
  CreateMLValue<float>(cpu_alloc, rotary_cache_dims, rotary_cache_data, &cos_cache_value);
  OrtValue sin_cache_value;
  CreateMLValue<float>(cpu_alloc, rotary_cache_dims, rotary_cache_data, &sin_cache_value);
  OrtValue output_value;
  CreateMLValue<float>(output_dims, output_data.data(), cpu_alloc->Info(), &output_value);

  std::unique_ptr<IOBinding> io_binding;
  ASSERT_STATUS_OK(session.NewIOBinding(&io_binding));
  ASSERT_STATUS_OK(io_binding->BindInput("query", query_value));
  ASSERT_STATUS_OK(io_binding->BindInput("key", key_value));
  ASSERT_STATUS_OK(io_binding->BindInput("value", value_value));
  ASSERT_STATUS_OK(io_binding->BindInput("past_key", past_key_value));
  ASSERT_STATUS_OK(io_binding->BindInput("past_value", past_value_value));
  ASSERT_STATUS_OK(io_binding->BindInput("block_row_indices", block_row_indices_value));
  ASSERT_STATUS_OK(io_binding->BindInput("block_col_indices", block_col_indices_value));
  ASSERT_STATUS_OK(io_binding->BindInput("total_sequence_length", total_sequence_length_value));
  ASSERT_STATUS_OK(io_binding->BindInput("key_total_sequence_lengths", total_key_lengths_value));
  ASSERT_STATUS_OK(io_binding->BindInput("cos_cache", cos_cache_value));
  ASSERT_STATUS_OK(io_binding->BindInput("sin_cache", sin_cache_value));
  ASSERT_STATUS_OK(io_binding->BindOutput("output", output_value));
  ASSERT_STATUS_OK(io_binding->BindOutput("present_key", past_key_value));
  ASSERT_STATUS_OK(io_binding->BindOutput("present_value", past_value_value));

  RunOptions run_options;
  ASSERT_STATUS_OK(session.Run(run_options, *io_binding));

  const auto& outputs = io_binding->GetOutputs();
  ASSERT_EQ(outputs.size(), 3u);
  EXPECT_EQ(outputs[1].Get<Tensor>().Data<float>(), past_key_data.data());
  EXPECT_EQ(outputs[2].Get<Tensor>().Data<float>(), past_value_data.data());

  for (float value : outputs[0].Get<Tensor>().DataAsSpan<float>()) {
    EXPECT_FLOAT_EQ(value, 0.0f);
  }
}

}  // namespace

TEST(SparseAttentionTest, RejectsOutOfRangeKeyTotalSequenceLengths) {
  RunSparseAttentionInvalidInputTest({-5}, {1}, "key_total_sequence_lengths value -5 at batch index 0 is out of range [1, 4]");
}

TEST(SparseAttentionTest, RejectsKeyTotalSequenceLengthsShapeMismatch) {
  RunSparseAttentionInvalidInputTest({4, 4}, {2}, "key_total_sequence_lengths must have shape (batch_size)");
}

TEST(SparseAttentionTest, RejectsPromptKeyTotalSequenceLengthsShorterThanSequenceLength) {
  RunSparseAttentionInvalidInputTest({0}, {1},
                                     "key_total_sequence_lengths value 0 at batch index 0 is out of range [1, 1]",
                                     1);
}

TEST(SparseAttentionTest, AcceptsPromptKeyTotalSequenceLengthsForPaddedBatch) {
  RunSparseAttentionPromptInputTest({5, 2}, 2, 5, 5);
}

}  // namespace test
}  // namespace onnxruntime
