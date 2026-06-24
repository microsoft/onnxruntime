// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Shape-inference correctness coverage for attention contrib ops with optional "present" outputs.
// DecoderAttention, MultiHeadAttention and DecoderMaskedMultiHeadAttention each expose an optional
// present_key (output 1) and present_value (output 2). These outputs are produced as a pair, so a
// node may declare either one output, or all three; declaring exactly two (present_key kept,
// present_value omitted) is also valid per the schemas.
//
// The "...Omitted" tests build each op with exactly two outputs and assert that Graph::Resolve()
// completes cleanly without referencing the absent third output. The "...AllPresentOutputs" tests
// build each op with all three outputs and assert that the present_key / present_value branch still
// runs and propagates their element types. Together they pin the guard to exactly "> 2": fewer
// outputs must not touch the missing one, and three outputs must still be inferred.
//
// These tests exercise only graph-load shape inference, which is execution-provider independent, so
// they run on the default CPU build with no provider-specific handling. The resolve path for these
// models is throw-free, so the tests are valid in builds compiled without exceptions
// (ORT_NO_EXCEPTIONS) and need no exception-specific guarding.

#include "gtest/gtest.h"

#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "test/test_environment.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {

namespace {

constexpr int kOnnxOpsetVersion = 17;

// Builds a single-node model via add_node, resolves it (running the node's type/shape inference),
// asserts success, and runs the optional verifier against the resolved graph.
void BuildResolveAndVerify(const std::function<void(ModelTestBuilder& builder)>& add_node,
                           const std::function<void(const Graph& graph)>& verify = nullptr) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = kOnnxOpsetVersion;
  domain_to_version[kMSDomain] = 1;

  Model model("attention_optional_outputs", /*is_onnx_domain_only=*/false, ModelMetaData(),
              PathString(), IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
              DefaultLoggingManager().DefaultLogger());

  ModelTestBuilder builder(model.MainGraph());
  add_node(builder);
  builder.SetGraphOutputs();

  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  if (verify) {
    verify(model.MainGraph());
  }
}

// Asserts that the given output NodeArg received a tensor element type from shape inference.
void ExpectInferredElemType(const NodeArg* output) {
  ASSERT_NE(output, nullptr);
  const ONNX_NAMESPACE::TypeProto* type = output->TypeAsProto();
  ASSERT_NE(type, nullptr);
  EXPECT_TRUE(type->has_tensor_type());
  EXPECT_TRUE(type->tensor_type().has_elem_type());
}

}  // namespace

// MultiHeadAttention with present_key kept and present_value omitted (exactly two outputs).
// past_key (input 6), past_value (input 7) and past_sequence_length (input 8) are supplied with
// shapes so the present-output branch is active; with only two outputs declared, inference must not
// reference the absent present_value (output 2). Inputs 1-5 are optional and left empty.
TEST(AttentionOptionalOutputsShapeInferenceTest, MultiHeadAttentionPresentValueOmitted) {
  BuildResolveAndVerify([](ModelTestBuilder& builder) {
    NodeArg& empty = builder.graph_.GetOrCreateNodeArg("", nullptr);
    NodeArg* query = builder.MakeInput<float>(std::vector<int64_t>{2, 1, 4});
    NodeArg* past_key = builder.MakeInput<float>(std::vector<int64_t>{2, 2, 3, 2});
    NodeArg* past_value = builder.MakeInput<float>(std::vector<int64_t>{2, 2, 3, 2});
    NodeArg* past_sequence_length = builder.MakeInput<int32_t>(std::vector<int64_t>{1});
    NodeArg* output = builder.MakeOutput<float>(std::nullopt);
    NodeArg* present_key = builder.MakeOutput<float>(std::nullopt);
    std::vector<NodeArg*> inputs = {query, &empty, &empty, &empty, &empty, &empty,
                                    past_key, past_value, past_sequence_length};
    Node& node = builder.AddNode("MultiHeadAttention", inputs, {output, present_key}, kMSDomain);
    node.AddAttribute("num_heads", static_cast<int64_t>(2));
  });
}

// DecoderMaskedMultiHeadAttention with present_key kept and present_value omitted.
// past_key (input 5) and past_value (input 6) are supplied with shapes and past buffer sharing is
// enabled so the present-output branch is active; with only two outputs declared, inference must not
// reference the absent present_value (output 2). Inputs 1-4 are optional and left empty.
TEST(AttentionOptionalOutputsShapeInferenceTest, DecoderMaskedMultiHeadAttentionPresentValueOmitted) {
  BuildResolveAndVerify([](ModelTestBuilder& builder) {
    NodeArg& empty = builder.graph_.GetOrCreateNodeArg("", nullptr);
    NodeArg* query = builder.MakeInput<float>(std::vector<int64_t>{2, 1, 4});
    NodeArg* past_key = builder.MakeInput<float>(std::vector<int64_t>{2, 2, 3, 2});
    NodeArg* past_value = builder.MakeInput<float>(std::vector<int64_t>{2, 2, 3, 2});
    NodeArg* output = builder.MakeOutput<float>(std::nullopt);
    NodeArg* present_key = builder.MakeOutput<float>(std::nullopt);
    std::vector<NodeArg*> inputs = {query, &empty, &empty, &empty, &empty, past_key, past_value};
    Node& node = builder.AddNode("DecoderMaskedMultiHeadAttention", inputs, {output, present_key},
                                 kMSDomain);
    node.AddAttribute("num_heads", static_cast<int64_t>(2));
    node.AddAttribute("past_present_share_buffer", static_cast<int64_t>(1));
  });
}

// DecoderAttention with new_key_cache kept and new_value_cache omitted (exactly two outputs).
TEST(AttentionOptionalOutputsShapeInferenceTest, DecoderAttentionNewValueCacheOmitted) {
  BuildResolveAndVerify([](ModelTestBuilder& builder) {
    // DecoderAttention requires inputs 0-4 and 8-11; inputs 5-7 are optional and left empty here.
    NodeArg& empty = builder.graph_.GetOrCreateNodeArg("", nullptr);
    NodeArg* query = builder.MakeInput<float>(std::vector<int64_t>{2, 1, 4});
    NodeArg* key = builder.MakeInput<float>(std::vector<int64_t>{2, 1, 4});
    NodeArg* q_weight = builder.MakeInput<float>(std::vector<int64_t>{4, 4});
    NodeArg* kv_weight = builder.MakeInput<float>(std::vector<int64_t>{4, 8});
    NodeArg* bias = builder.MakeInput<float>(std::vector<int64_t>{12});
    NodeArg* static_kv = builder.MakeInput<bool>(std::vector<int64_t>{1});
    NodeArg* use_past = builder.MakeInput<bool>(std::vector<int64_t>{1});
    NodeArg* has_layer_state = builder.MakeInput<bool>(std::vector<int64_t>{1});
    NodeArg* has_key_padding_mask = builder.MakeInput<bool>(std::vector<int64_t>{1});

    NodeArg* output = builder.MakeOutput<float>(std::nullopt);
    NodeArg* new_key_cache = builder.MakeOutput<float>(std::nullopt);

    std::vector<NodeArg*> inputs = {query, key, q_weight, kv_weight, bias, &empty, &empty, &empty,
                                    static_kv, use_past, has_layer_state, has_key_padding_mask};
    Node& node = builder.AddNode("DecoderAttention", inputs, {output, new_key_cache}, kMSDomain);
    node.AddAttribute("num_heads", static_cast<int64_t>(2));
  });
}

// MultiHeadAttention with all three outputs: the present_key / present_value branch must still run.
TEST(AttentionOptionalOutputsShapeInferenceTest, MultiHeadAttentionAllPresentOutputs) {
  NodeArg* present_key = nullptr;
  NodeArg* present_value = nullptr;
  BuildResolveAndVerify(
      [&](ModelTestBuilder& builder) {
        // present_key/present_value types are propagated from past_key (input 6) and past_value
        // (input 7) when past buffer sharing is active, which the op detects from shaped past_key
        // (input 6) and past_sequence_length (input 8). Leave inputs 1-5 empty to reach them.
        NodeArg& empty = builder.graph_.GetOrCreateNodeArg("", nullptr);
        NodeArg* query = builder.MakeInput<float>(std::vector<int64_t>{2, 1, 4});
        NodeArg* past_key = builder.MakeInput<float>(std::vector<int64_t>{2, 2, 3, 2});
        NodeArg* past_value = builder.MakeInput<float>(std::vector<int64_t>{2, 2, 3, 2});
        NodeArg* past_sequence_length = builder.MakeInput<int32_t>(std::vector<int64_t>{1});

        NodeArg* output = builder.MakeOutput<float>(std::nullopt);
        present_key = builder.MakeOutput();
        present_value = builder.MakeOutput();

        std::vector<NodeArg*> inputs = {query, &empty, &empty, &empty, &empty, &empty,
                                        past_key, past_value, past_sequence_length};
        Node& node = builder.AddNode("MultiHeadAttention", inputs,
                                     {output, present_key, present_value}, kMSDomain);
        node.AddAttribute("num_heads", static_cast<int64_t>(2));
      },
      [&](const Graph&) {
        ExpectInferredElemType(present_key);
        ExpectInferredElemType(present_value);
      });
}

// DecoderMaskedMultiHeadAttention with all three outputs: shape inference must still populate them.
TEST(AttentionOptionalOutputsShapeInferenceTest, DecoderMaskedMultiHeadAttentionAllPresentOutputs) {
  NodeArg* present_key = nullptr;
  NodeArg* present_value = nullptr;
  BuildResolveAndVerify(
      [&](ModelTestBuilder& builder) {
        // For this op past_key/past_value are inputs 5 and 6; leave inputs 1-4 empty to reach them.
        NodeArg& empty = builder.graph_.GetOrCreateNodeArg("", nullptr);
        NodeArg* query = builder.MakeInput<float>(std::vector<int64_t>{2, 1, 4});
        NodeArg* past_key = builder.MakeInput<float>(std::vector<int64_t>{2, 2, 3, 2});
        NodeArg* past_value = builder.MakeInput<float>(std::vector<int64_t>{2, 2, 3, 2});

        NodeArg* output = builder.MakeOutput<float>(std::nullopt);
        present_key = builder.MakeOutput();
        present_value = builder.MakeOutput();

        std::vector<NodeArg*> inputs = {query, &empty, &empty, &empty, &empty, past_key, past_value};
        Node& node = builder.AddNode("DecoderMaskedMultiHeadAttention", inputs,
                                     {output, present_key, present_value}, kMSDomain);
        node.AddAttribute("num_heads", static_cast<int64_t>(2));
        node.AddAttribute("past_present_share_buffer", static_cast<int64_t>(1));
      },
      [&](const Graph&) {
        ExpectInferredElemType(present_key);
        ExpectInferredElemType(present_value);
      });
}

// DecoderAttention with all three outputs: new_key_cache / new_value_cache types must be inferred.
TEST(AttentionOptionalOutputsShapeInferenceTest, DecoderAttentionAllCacheOutputs) {
  NodeArg* new_key_cache = nullptr;
  NodeArg* new_value_cache = nullptr;
  BuildResolveAndVerify(
      [&](ModelTestBuilder& builder) {
        NodeArg& empty = builder.graph_.GetOrCreateNodeArg("", nullptr);
        NodeArg* query = builder.MakeInput<float>(std::vector<int64_t>{2, 1, 4});
        NodeArg* key = builder.MakeInput<float>(std::vector<int64_t>{2, 1, 4});
        NodeArg* q_weight = builder.MakeInput<float>(std::vector<int64_t>{4, 4});
        NodeArg* kv_weight = builder.MakeInput<float>(std::vector<int64_t>{4, 8});
        NodeArg* bias = builder.MakeInput<float>(std::vector<int64_t>{12});
        NodeArg* key_cache = builder.MakeInput<float>(std::vector<int64_t>{2, 2, 3, 2});
        NodeArg* value_cache = builder.MakeInput<float>(std::vector<int64_t>{2, 2, 3, 2});
        NodeArg* static_kv = builder.MakeInput<bool>(std::vector<int64_t>{1});
        NodeArg* use_past = builder.MakeInput<bool>(std::vector<int64_t>{1});
        NodeArg* has_layer_state = builder.MakeInput<bool>(std::vector<int64_t>{1});
        NodeArg* has_key_padding_mask = builder.MakeInput<bool>(std::vector<int64_t>{1});

        NodeArg* output = builder.MakeOutput<float>(std::nullopt);
        new_key_cache = builder.MakeOutput();
        new_value_cache = builder.MakeOutput();

        std::vector<NodeArg*> inputs = {query, key, q_weight, kv_weight, bias, &empty, key_cache,
                                        value_cache, static_kv, use_past, has_layer_state,
                                        has_key_padding_mask};
        Node& node = builder.AddNode("DecoderAttention", inputs,
                                     {output, new_key_cache, new_value_cache}, kMSDomain);
        node.AddAttribute("num_heads", static_cast<int64_t>(2));
      },
      [&](const Graph&) {
        ExpectInferredElemType(new_key_cache);
        ExpectInferredElemType(new_value_cache);
      });
}

}  // namespace test
}  // namespace onnxruntime
