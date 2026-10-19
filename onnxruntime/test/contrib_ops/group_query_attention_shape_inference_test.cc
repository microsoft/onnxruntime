// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "gtest/gtest.h"

#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "test/test_environment.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {

common::Status ResolveGroupQueryAttention(
    int64_t num_heads,
    int64_t kv_num_heads,
    const std::vector<std::variant<int64_t, std::string>>& query_shape,
    const std::function<void(const NodeArg&)>& verify = nullptr,
    bool provide_unshaped_value = false,
    const std::function<void(const NodeArg&, const NodeArg&)>& verify_present = nullptr) {
  const std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 17}, {kMSDomain, 1}};
  Model model("packed_group_query_attention", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
              DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  NodeArg& empty = graph.GetOrCreateNodeArg("", nullptr);
  NodeArg* query = builder.MakeSymbolicInput<float>(query_shape);
  NodeArg* key = &empty;
  NodeArg* value = &empty;
  if (provide_unshaped_value) {
    key = builder.MakeInput<float>(std::vector<int64_t>{1, 1, 8});
    value = builder.MakeInput<float>(std::nullopt);
  }
  NodeArg* seqlens_k = builder.MakeInput<int32_t>({1}, {0});
  NodeArg* total_sequence_length = builder.MakeInput<int32_t>({1}, {1});
  NodeArg* output = builder.MakeOutput();
  std::vector<NodeArg*> outputs{output};
  NodeArg* present_key = nullptr;
  NodeArg* present_value = nullptr;
  if (verify_present) {
    present_key = builder.MakeOutput();
    present_value = builder.MakeOutput();
    outputs.push_back(present_key);
    outputs.push_back(present_value);
  }
  std::vector<NodeArg*> inputs = {query, key, value, &empty, &empty,
                                  seqlens_k, total_sequence_length};
  Node& node = builder.AddNode("GroupQueryAttention", inputs, outputs, kMSDomain);
  node.AddAttribute("num_heads", num_heads);
  node.AddAttribute("kv_num_heads", kv_num_heads);
  builder.SetGraphOutputs();

  auto status = graph.Resolve();
  if (status.IsOK() && verify) {
    verify(*output);
  }
  if (status.IsOK() && verify_present) {
    verify_present(*present_key, *present_value);
  }
  return status;
}

#ifndef ORT_NO_EXCEPTIONS

void ExpectResolveFailure(int64_t num_heads,
                          int64_t kv_num_heads,
                          int64_t hidden_size,
                          const std::string& expected_message) {
  auto status = ResolveGroupQueryAttention(
      num_heads, kv_num_heads, {int64_t{1}, int64_t{1}, hidden_size});
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find(expected_message), std::string::npos)
      << status.ErrorMessage();
}

TEST(GroupQueryAttentionShapeInferenceTest, RejectsNonPositiveHeadCounts) {
  for (int64_t invalid_num_heads : {int64_t{0}, int64_t{-1}}) {
    SCOPED_TRACE("num_heads=" + std::to_string(invalid_num_heads));
    ExpectResolveFailure(invalid_num_heads, 1, 3,
                         "num_heads and kv_num_heads must be positive.");
  }

  for (int64_t invalid_kv_num_heads : {int64_t{0}, int64_t{-1}}) {
    SCOPED_TRACE("kv_num_heads=" + std::to_string(invalid_kv_num_heads));
    ExpectResolveFailure(1, invalid_kv_num_heads, 3,
                         "num_heads and kv_num_heads must be positive.");
  }
}

TEST(GroupQueryAttentionShapeInferenceTest, RejectsGroupedHeadCountOverflow) {
  ExpectResolveFailure(std::numeric_limits<int64_t>::max(), 1, 3,
                       "num_heads + 2 * kv_num_heads must not overflow.");
}

TEST(GroupQueryAttentionShapeInferenceTest, RejectsNonDivisiblePackedHiddenSize) {
  ExpectResolveFailure(2, 1, 5,
                       "Packed query hidden size must be divisible by the grouped head count.");
}

#endif  // ORT_NO_EXCEPTIONS

TEST(GroupQueryAttentionShapeInferenceTest, LeavesDerivedSymbolicHiddenSizeUnknown) {
  auto status = ResolveGroupQueryAttention(
      2, 1, {int64_t{1}, int64_t{2}, std::string{"packed_hidden_size"}},
      [](const NodeArg& output) {
        const auto* shape = output.Shape();
        ASSERT_NE(shape, nullptr);
        ASSERT_EQ(shape->dim_size(), 3);
        EXPECT_EQ(shape->dim(0).dim_value(), 1);
        EXPECT_EQ(shape->dim(1).dim_value(), 2);
        EXPECT_FALSE(shape->dim(2).has_dim_value());
        EXPECT_FALSE(shape->dim(2).has_dim_param());
      });
  ASSERT_STATUS_OK(status);
}

TEST(GroupQueryAttentionShapeInferenceTest, UnshapedValueInputIsNotPacked) {
  auto status = ResolveGroupQueryAttention(
      4, 1, {int64_t{1}, int64_t{1}, int64_t{32}},
      [](const NodeArg& output) {
        const auto* shape = output.Shape();
        ASSERT_NE(shape, nullptr);
        ASSERT_EQ(shape->dim_size(), 3);
        EXPECT_EQ(shape->dim(0).dim_value(), 1);
        EXPECT_EQ(shape->dim(1).dim_value(), 1);
        EXPECT_EQ(shape->dim(2).dim_value(), 32);
      },
      true,
      [](const NodeArg& present_key, const NodeArg& present_value) {
        auto verify_present = [](const NodeArg& present) {
          const auto* shape = present.Shape();
          ASSERT_NE(shape, nullptr);
          ASSERT_EQ(shape->dim_size(), 4);
          EXPECT_EQ(shape->dim(3).dim_value(), 8);
        };
        verify_present(present_key);
        verify_present(present_value);
      });
  ASSERT_STATUS_OK(status);
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
