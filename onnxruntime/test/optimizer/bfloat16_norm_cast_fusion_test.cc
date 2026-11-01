// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>
#include <limits>

#include "core/graph/graph_utils.h"
#include "core/optimizer/bfloat16_norm_cast_fusion.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/unittest_util/conversion.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
#if !defined(DISABLE_CONTRIB_OPS)
namespace {

struct NormCastOptions {
  bool leading = true;
  bool trailing = true;
  bool dynamic_scale = false;
  bool overridable_scale = false;
  bool shared_scale = false;
  bool shared_leading = false;
  bool shared_norm = false;
  bool exposed_leading = false;
  bool exposed_norm = false;
  bool auxiliary_output = false;
  bool cpu_norm = false;
  bool cpu_cast = false;
  bool fp16_input = false;
  bool produced_input = false;
  bool shared_trailing = false;
  std::vector<float> scale{0.125f, -0.75f, 1.5f, -0.0f};
};

void BuildNormCast(ModelTestBuilder& builder, const NormCastOptions& options) {
  const auto width = static_cast<int64_t>(options.scale.size());
  const std::vector<int64_t> shape{2, width};
  std::vector<float> values(2 * options.scale.size());
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = i % 2 == 0 ? -1.0f : 1.0f;
  }
  NodeArg* input = options.leading
                       ? (options.fp16_input
                              ? builder.MakeInput<MLFloat16>(shape, FloatsToMLFloat16s(values))
                              : builder.MakeInput<BFloat16>(shape, FloatsToBFloat16s(values)))
                       : builder.MakeInput<float>(shape, values);
  if (options.produced_input) {
    auto* produced = builder.MakeIntermediate<BFloat16>(shape);
    auto& producer = builder.AddNode("Identity", {input}, {produced});
    producer.SetExecutionProviderType(kCudaExecutionProvider);
    input = produced;
  }
  if (options.leading) {
    auto* cast_output = builder.MakeIntermediate<float>(shape);
    auto& cast = builder.AddNode("Cast", {input}, {cast_output});
    cast.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto::FLOAT});
    cast.SetExecutionProviderType(options.cpu_cast ? kCpuExecutionProvider : kCudaExecutionProvider);
    input = cast_output;
    if (options.shared_leading) {
      builder.AddNode("Identity", {input}, {builder.MakeOutput<float>(shape)});
    }
    if (options.exposed_leading) {
      builder.output_names_.push_back(input->Name());
    }
  }
  auto* scale = options.dynamic_scale || options.overridable_scale
                    ? builder.MakeInput<float>({width}, options.scale)
                    : builder.MakeInitializer<float>({width}, options.scale);
  if (options.overridable_scale) {
    ONNX_NAMESPACE::TensorProto tensor;
    tensor.set_name(scale->Name());
    tensor.set_data_type(ONNX_NAMESPACE::TensorProto::FLOAT);
    tensor.add_dims(width);
    for (const float value : options.scale) {
      tensor.add_float_data(value);
    }
    builder.graph_.AddInitializedTensor(tensor);
    std::vector<const NodeArg*> graph_inputs;
    for (const auto& feed : builder.feeds_) {
      graph_inputs.push_back(builder.graph_.GetNodeArg(feed.first));
    }
    builder.graph_.SetInputs(graph_inputs);
  }
  if (options.shared_scale) {
    builder.AddNode("Identity", {scale}, {builder.MakeOutput<float>(std::vector<int64_t>{width})});
  }
  auto* normalized = options.trailing ? builder.MakeIntermediate<float>(shape) : builder.MakeOutput<float>(shape);
  std::vector<NodeArg*> outputs{normalized};
  if (options.auxiliary_output) {
    outputs.push_back(builder.MakeOutput<float>(std::vector<int64_t>{2, 1}));
  }
  auto& norm = builder.AddNode("SimplifiedLayerNormalization", {input, scale}, outputs);
  norm.AddAttribute("epsilon", 1e-5f);
  norm.SetExecutionProviderType(options.cpu_norm ? kCpuExecutionProvider : kCudaExecutionProvider);
  if (options.shared_norm) {
    builder.AddNode("Identity", {normalized}, {builder.MakeOutput<float>(shape)});
  }
  if (options.exposed_norm) {
    builder.output_names_.push_back(normalized->Name());
  }
  if (options.trailing) {
    auto* output = options.shared_trailing ? builder.MakeIntermediate<BFloat16>(shape) : builder.MakeOutput<BFloat16>(shape);
    auto& cast = builder.AddNode("Cast", {normalized}, {output});
    cast.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto::BFLOAT16});
    cast.SetExecutionProviderType(kCudaExecutionProvider);
    if (options.shared_trailing) {
      builder.AddNode("Identity", {output}, {builder.MakeOutput<BFloat16>(shape)});
      builder.AddNode("Identity", {output}, {builder.MakeOutput<BFloat16>(shape)});
    }
  }
}

Status CheckCastCount(const Graph& graph, int expected) {
  const auto counts = CountOpsInGraph(graph);
  ORT_RETURN_IF_NOT(counts.at("SimplifiedLayerNormalization") == 1, "Normalization was removed.");
  const auto casts = counts.find("Cast");
  ORT_RETURN_IF_NOT((casts == counts.end() ? 0 : casts->second) == expected, "Unexpected remaining Cast count.");
  return Status::OK();
}

}  // namespace

TEST_F(GraphTransformationTests, BFloat16NormCastFusionGraphSafety) {
  struct Case {
    NormCastOptions options;
    int casts;
  };
  std::vector<Case> cases{{{}, 0}};
  NormCastOptions options;
  options.trailing = false;
  cases.push_back({options, 0});
  options = {};
  options.leading = false;
  cases.push_back({options, 0});
  options = {};
  options.scale[0] = 1.0001f;
  cases.push_back({options, 1});
  for (const float value : {std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()}) {
    options = {};
    options.scale[0] = value;
    cases.push_back({options, 1});
  }
  options = {};
  options.dynamic_scale = true;
  cases.push_back({options, 1});
  options = {};
  options.overridable_scale = true;
  cases.push_back({options, 1});
  options = {};
  options.shared_scale = true;
  cases.push_back({options, 0});
  options = {};
  options.shared_leading = true;
  cases.push_back({options, 1});
  options = {};
  options.exposed_leading = true;
  cases.push_back({options, 1});
  options = {};
  options.shared_norm = true;
  cases.push_back({options, 1});
  options = {};
  options.exposed_norm = true;
  cases.push_back({options, 1});
  options = {};
  options.auxiliary_output = true;
  cases.push_back({options, 2});
  options = {};
  options.cpu_norm = true;
  cases.push_back({options, 2});
  options = {};
  options.cpu_cast = true;
  cases.push_back({options, 1});
  options = {};
  options.fp16_input = true;
  cases.push_back({options, 1});
  options = {};
  options.produced_input = true;
  cases.push_back({options, 0});
  options = {};
  options.shared_trailing = true;
  cases.push_back({options, 0});

  for (size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(i);
    ASSERT_STATUS_OK(TestGraphTransformer(
        [&](ModelTestBuilder& builder) { BuildNormCast(builder, cases[i].options); }, 14, *logger_,
        std::make_unique<BFloat16NormCastFusion>(), TransformerLevel::Level2, 2, {},
        [&](Graph& graph) { return CheckCastCount(graph, cases[i].casts); }));
  }
}

TEST_F(GraphTransformationTests, BFloat16NormCastFusionPreservesSharedScaleBits) {
  NormCastOptions options;
  options.shared_scale = true;
  ASSERT_STATUS_OK(TestGraphTransformer(
      [&](ModelTestBuilder& builder) { BuildNormCast(builder, options); }, 14, *logger_,
      std::make_unique<BFloat16NormCastFusion>(), TransformerLevel::Level2, 2, {},
      [&](Graph& graph) {
        ORT_RETURN_IF_ERROR(CheckCastCount(graph, 0));
        for (const auto& node : graph.Nodes()) {
          const ONNX_NAMESPACE::TensorProto* tensor = nullptr;
          ORT_RETURN_IF_NOT(graph.GetInitializedTensor(node.InputDefs()[node.OpType() == "Identity" ? 0 : 1]->Name(), tensor),
                            "Scale initializer missing.");
          Initializer scale(graph, *tensor, graph.ModelPath());
          if (node.OpType() == "Identity") {
            ORT_RETURN_IF_NOT(scale.data_type() == ONNX_NAMESPACE::TensorProto::FLOAT, "Shared FP32 scale changed.");
            ORT_RETURN_IF_NOT(std::memcmp(scale.data<float>(), options.scale.data(), 4 * sizeof(float)) == 0,
                              "Shared scale bits changed.");
          } else {
            ORT_RETURN_IF_NOT(scale.data_type() == ONNX_NAMESPACE::TensorProto::BFLOAT16, "Scale not converted.");
            for (size_t i = 0; i < 4; ++i) {
              const float restored = scale.data<BFloat16>()[i].ToFloat();
              ORT_RETURN_IF_NOT(std::memcmp(&restored, &options.scale[i], sizeof(float)) == 0,
                                "BF16 scale conversion changed bits.");
            }
          }
        }
        return Status::OK();
      }));
}

TEST_F(GraphTransformationTests, BFloat16NormCastFusionOptInRegistration) {
  auto cpu = DefaultCpuExecutionProvider();
  ASSERT_NE(cpu, nullptr);
  for (const std::string value : {"", "0", "1"}) {
    SessionOptions options;
    if (!value.empty()) {
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsEnableBFloat16NormCastFusion, value.c_str()));
    }
    auto transformers = optimizer_utils::GenerateTransformers(TransformerLevel::Level2, options, *cpu, *logger_);
    bool found = false;
    for (const auto& transformer : transformers) {
      if (transformer->Name() == "BFloat16NormCastFusion") {
        found = true;
      }
      if (transformer->Name() == "GroupQueryAttentionPreNormFusion") {
        EXPECT_EQ(transformer->GetCompatibleExecutionProviders().count(kCudaExecutionProvider), value == "1" ? 0u : 1u);
        EXPECT_EQ(transformer->GetCompatibleExecutionProviders().count(kWebGpuExecutionProvider), 1u);
      }
    }
    EXPECT_EQ(found, value == "1");
  }
  SessionOptions invalid;
  ASSERT_STATUS_OK(invalid.config_options.AddConfigEntry(kOrtSessionOptionsEnableBFloat16NormCastFusion, "invalid"));
  EXPECT_THROW(optimizer_utils::GenerateTransformers(TransformerLevel::Level2, invalid, *cpu, *logger_), OnnxRuntimeException);
}

TEST_F(GraphTransformationTests, BFloat16NormCastFusionExternalInitializer) {
  NormCastOptions options;
  options.scale.assign(64, 1.0f);
  ASSERT_STATUS_OK(TestGraphTransformer(
      [&](ModelTestBuilder& builder) { BuildNormCast(builder, options); }, 14, *logger_,
      std::make_unique<BFloat16NormCastFusion>(), TransformerLevel::Level2, 2,
      [](Graph& graph) {
        ORT_RETURN_IF_ERROR(graph.ConvertInitializersIntoOrtValues());
        for (const auto& node : graph.Nodes()) {
          if (node.OpType() == "SimplifiedLayerNormalization") {
            const auto* tensor = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
            ORT_RETURN_IF_NOT(tensor != nullptr && utils::HasExternalDataInMemory(*tensor),
                              "Test scale must have in-memory external data.");
          }
        }
        return Status::OK();
      },
      [](Graph& graph) { return CheckCastCount(graph, 0); }));
}

#ifdef USE_CUDA
TEST_F(GraphTransformationTests, BFloat16NormCastFusionCudaNumericalParity) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device with compute capability 8.0 or newer is required.";
  }
  for (const bool leading : {false, true}) {
    for (const bool trailing : {false, true}) {
      if (!leading && !trailing) {
        continue;
      }
      NormCastOptions options;
      options.leading = leading;
      options.trailing = trailing;
      TransformerTester(
          [&](ModelTestBuilder& builder) { BuildNormCast(builder, options); },
          [](InferenceSessionWrapper& session) { ASSERT_STATUS_OK(CheckCastCount(session.GetGraph(), 0)); },
          TransformerLevel::Level1, TransformerLevel::Level2, 14, 0.0, 0.0, nullptr,
          [](SessionOptions& session_options) {
            ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsEnableBFloat16NormCastFusion, "1"));
          },
          {}, DefaultCudaExecutionProvider());
    }
  }
}
#endif
#endif
}  // namespace test
}  // namespace onnxruntime
