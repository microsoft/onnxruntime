// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstring>
#include <limits>

#include "core/framework/data_transfer.h"
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
  std::vector<float> input_values;
};

void BuildNormCast(ModelTestBuilder& builder, const NormCastOptions& options) {
  const auto width = static_cast<int64_t>(options.scale.size());
  const std::vector<int64_t> shape{2, width};
  std::vector<float> values(2 * options.scale.size());
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = i % 2 == 0 ? -1.0f : 1.0f;
  }
  if (!options.input_values.empty()) {
    ORT_ENFORCE(options.input_values.size() == values.size());
    values = options.input_values;
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

template <typename T>
bool SameNonNaNBits(const T& reference, const T& actual) {
  const bool reference_nan = std::isnan(static_cast<float>(reference));
  const bool actual_nan = std::isnan(static_cast<float>(actual));
  return reference_nan || actual_nan
             ? reference_nan == actual_nan
             : std::memcmp(&reference, &actual, sizeof(T)) == 0;
}

}  // namespace

TEST_F(GraphTransformationTests, BFloat16NormCastFusionBitwiseComparison) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(SameNonNaNBits(0.0f, -0.0f));
  EXPECT_FALSE(SameNonNaNBits(BFloat16(0.0f), BFloat16(-0.0f)));
  EXPECT_TRUE(SameNonNaNBits(-0.0f, -0.0f));
  EXPECT_TRUE(SameNonNaNBits(BFloat16(-0.0f), BFloat16(-0.0f)));
  EXPECT_TRUE(SameNonNaNBits(nan, -nan));
  EXPECT_TRUE(SameNonNaNBits(BFloat16(nan), BFloat16(-nan)));
  EXPECT_FALSE(SameNonNaNBits(nan, 1.0f));
  EXPECT_FALSE(SameNonNaNBits(1.0f, nan));
  EXPECT_FALSE(SameNonNaNBits(BFloat16(nan), BFloat16(1.0f)));
  EXPECT_FALSE(SameNonNaNBits(BFloat16(1.0f), BFloat16(nan)));
}

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

TEST_F(GraphTransformationTests, BFloat16NormCastFusionExcludesSessionScale) {
  for (const bool leading : {false, true}) {
    Model model("ExcludedNormScale", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 14}}, {}, *logger_);
    ModelTestBuilder builder(model.MainGraph());
    NormCastOptions options;
    options.leading = leading;
    BuildNormCast(builder, options);
    builder.SetGraphOutputs();
    ASSERT_STATUS_OK(model.MainGraph().Resolve());
    std::string scale_name;
    for (const auto& node : model.MainGraph().Nodes()) {
      if (node.OpType() == "SimplifiedLayerNormalization") {
        scale_name = node.InputDefs()[1]->Name();
      }
    }
    ASSERT_FALSE(scale_name.empty());
    BFloat16NormCastFusion transformer({scale_name});
    bool modified = false;
    ASSERT_STATUS_OK(transformer.Apply(model.MainGraph(), modified, *logger_));
    EXPECT_EQ(modified, leading);
    ASSERT_STATUS_OK(model.MainGraph().Resolve());
    ASSERT_STATUS_OK(CheckCastCount(model.MainGraph(), 1));
    for (const auto& node : model.MainGraph().Nodes()) {
      if (node.OpType() == "SimplifiedLayerNormalization") {
        EXPECT_EQ(node.InputDefs()[1]->Name(), scale_name);
        EXPECT_EQ(node.InputDefs()[1]->TypeAsProto()->tensor_type().elem_type(), ONNX_NAMESPACE::TensorProto::FLOAT);
      }
    }
  }
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
      options.scale = {0.125f, -0.75f, 0.0f, -0.0f};
      struct Case {
        bool nan_input;
        bool shared_initializer;
      };
      for (const auto& test_case : {Case{false, false}, {true, false}, {false, true}, {true, true}}) {
        const bool nan_input = test_case.nan_input;
        const bool shared_initializer = test_case.shared_initializer;
        SCOPED_TRACE(testing::Message() << "leading=" << leading << " trailing=" << trailing
                                        << " nan_input=" << nan_input << " shared_initializer=" << shared_initializer);
        options.input_values = {-1, 1, -1, 1, 1, -1, -1, -1};
        if (nan_input) {
          options.input_values[0] = std::numeric_limits<float>::quiet_NaN();
        }
        Model model("NormCastBitwiseParity", false, ModelMetaData(), PathString(),
                    IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 14}}, {}, *logger_);
        ModelTestBuilder builder(model.MainGraph());
        BuildNormCast(builder, options);
        builder.SetGraphOutputs();
        ASSERT_STATUS_OK(model.MainGraph().Resolve());
        std::string scale_name;
        for (const auto& node : model.MainGraph().Nodes()) {
          if (node.OpType() == "SimplifiedLayerNormalization") {
            scale_name = node.InputDefs()[1]->Name();
          }
        }
        ASSERT_FALSE(scale_name.empty());
        std::string model_data;
        ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));

        auto run = [&](bool fused, std::vector<OrtValue>& fetches) {
          SessionOptions session_options;
          session_options.graph_optimization_level = TransformerLevel::Level2;
          ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
              kOrtSessionOptionsEnableBFloat16NormCastFusion, fused ? "1" : "0"));
          auto cuda_ep = DefaultCudaExecutionProvider();
          IAllocatorUniquePtr<float> scale_buffer;
          OrtValue shared_scale;
          if (shared_initializer) {
            AllocatorPtr gpu_allocator;
            for (const auto& allocator : cuda_ep->CreatePreferredAllocators()) {
              if (allocator->Info().device.Type() == OrtDevice::GPU &&
                  allocator->Info().mem_type == OrtMemTypeDefault) {
                gpu_allocator = allocator;
                break;
              }
            }
            ASSERT_NE(gpu_allocator, nullptr);
            auto values = options.scale;
            for (auto& value : values) {
              value *= 2.0f;
            }
            const TensorShape shape({static_cast<int64_t>(values.size())});
            const auto type = DataTypeImpl::GetType<float>();
            scale_buffer = IAllocator::MakeUniquePtr<float>(gpu_allocator, values.size());
            Tensor::InitOrtValue(type, shape, scale_buffer.get(), gpu_allocator->Info(), shared_scale);
            Tensor host_scale(type, shape, values.data(), OrtMemoryInfo());
            const auto transfer = cuda_ep->GetDataTransfer();
            ASSERT_NE(transfer, nullptr);
            ASSERT_STATUS_OK(transfer->CopyTensor(host_scale, *shared_scale.GetMutable<Tensor>()));
            ASSERT_STATUS_OK(session_options.AddInitializer(scale_name.c_str(), &shared_scale));
          }
          InferenceSessionWrapper session{session_options, GetEnvironment()};
          ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(cuda_ep)));
          ASSERT_STATUS_OK(session.Load(model_data.data(), static_cast<int>(model_data.size())));
          ASSERT_STATUS_OK(session.Initialize());
          EXPECT_STATUS_OK(CheckCastCount(session.GetGraph(), fused ? static_cast<int>(shared_initializer && trailing)
                                                                    : static_cast<int>(leading) + static_cast<int>(trailing)));
          ASSERT_STATUS_OK(session.Run(RunOptions(), builder.feeds_, builder.output_names_, &fetches));
        };
        std::vector<OrtValue> reference;
        std::vector<OrtValue> actual;
        ASSERT_NO_FATAL_FAILURE(run(false, reference));
        ASSERT_NO_FATAL_FAILURE(run(true, actual));
        ASSERT_EQ(reference.size(), 1u);
        ASSERT_EQ(actual.size(), 1u);
        const auto& reference_tensor = reference[0].Get<Tensor>();
        const auto& actual_tensor = actual[0].Get<Tensor>();
        ASSERT_EQ(reference_tensor.Shape(), actual_tensor.Shape());
        ASSERT_EQ(reference_tensor.GetElementType(), actual_tensor.GetElementType());
        size_t nan_count = 0;
        size_t positive_zeros = 0;
        size_t negative_zeros = 0;
        auto compare = [&]<typename T>() {
          const auto reference_values = reference_tensor.DataAsSpan<T>();
          const auto actual_values = actual_tensor.DataAsSpan<T>();
          ASSERT_EQ(reference_values.size(), actual_values.size());
          if (shared_initializer) {
            // This finite row must use the supplied scale, not the model's original 0.125 scale.
            EXPECT_GT(static_cast<float>(reference_values[4]), 0.24f);
            EXPECT_LT(static_cast<float>(reference_values[4]), 0.26f);
          }
          for (size_t i = 0; i < reference_values.size(); ++i) {
            EXPECT_TRUE(SameNonNaNBits(reference_values[i], actual_values[i])) << "element " << i;
            const float value = static_cast<float>(reference_values[i]);
            nan_count += std::isnan(value);
            positive_zeros += value == 0.0f && !std::signbit(value);
            negative_zeros += value == 0.0f && std::signbit(value);
          }
        };
        if (trailing) {
          ASSERT_TRUE(reference_tensor.IsDataType<BFloat16>());
          ASSERT_NO_FATAL_FAILURE(compare.template operator()<BFloat16>());
        } else {
          ASSERT_TRUE(reference_tensor.IsDataType<float>());
          ASSERT_NO_FATAL_FAILURE(compare.template operator()<float>());
        }
        EXPECT_EQ(nan_count, nan_input ? 4u : 0u);
        EXPECT_GT(positive_zeros, 0u);
        EXPECT_GT(negative_zeros, 0u);
      }
    }
  }
}
#endif
#endif
}  // namespace test
}  // namespace onnxruntime
