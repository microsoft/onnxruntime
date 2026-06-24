// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Regression coverage for attention contrib ops declared with their optional "present" outputs
// partially omitted. DecoderAttention, MultiHeadAttention and DecoderMaskedMultiHeadAttention each
// expose an optional present_key (output 1) and present_value (output 2). A node that declares
// exactly two outputs (present_key kept, present_value omitted) is valid per the op schemas, and
// graph resolution / shape inference must complete cleanly without referencing the absent third
// output.
//
// These tests build each op with exactly two outputs and assert that Graph::Resolve() succeeds.
// They exercise only graph-load shape inference, which is execution-provider independent, so they
// run on the default CPU build with no provider-specific handling. The resolve path for these
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

// Builds a single-node model via the supplied callback and verifies that graph resolution, which
// runs the node's type/shape inference, completes without error.
void ResolveSingleNodeModel(const std::function<void(ModelTestBuilder& builder)>& add_node) {
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
}

}  // namespace

// MultiHeadAttention with present_key kept and present_value omitted (exactly two outputs).
TEST(AttentionOptionalOutputsShapeInferenceTest, MultiHeadAttentionPresentValueOmitted) {
  ResolveSingleNodeModel([](ModelTestBuilder& builder) {
    NodeArg* query = builder.MakeInput<float>(std::vector<int64_t>{2, 1, 4});
    NodeArg* output = builder.MakeOutput<float>(std::nullopt);
    NodeArg* present_key = builder.MakeOutput<float>(std::nullopt);
    Node& node = builder.AddNode("MultiHeadAttention", {query}, {output, present_key}, kMSDomain);
    node.AddAttribute("num_heads", static_cast<int64_t>(2));
  });
}

// DecoderMaskedMultiHeadAttention with present_key kept and present_value omitted.
TEST(AttentionOptionalOutputsShapeInferenceTest, DecoderMaskedMultiHeadAttentionPresentValueOmitted) {
  ResolveSingleNodeModel([](ModelTestBuilder& builder) {
    NodeArg* query = builder.MakeInput<float>(std::vector<int64_t>{2, 1, 4});
    NodeArg* output = builder.MakeOutput<float>(std::nullopt);
    NodeArg* present_key = builder.MakeOutput<float>(std::nullopt);
    Node& node = builder.AddNode("DecoderMaskedMultiHeadAttention", {query}, {output, present_key},
                                 kMSDomain);
    node.AddAttribute("num_heads", static_cast<int64_t>(2));
  });
}

// DecoderAttention with new_key_cache kept and new_value_cache omitted (exactly two outputs).
TEST(AttentionOptionalOutputsShapeInferenceTest, DecoderAttentionNewValueCacheOmitted) {
  ResolveSingleNodeModel([](ModelTestBuilder& builder) {
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

}  // namespace test
}  // namespace onnxruntime
