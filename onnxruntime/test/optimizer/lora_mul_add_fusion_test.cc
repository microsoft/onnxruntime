// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/matmul_nbits_fusion.h"
#include "test/test_environment.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"

#if !defined(DISABLE_CONTRIB_OPS) && !defined(ORT_MINIMAL_BUILD)
namespace onnxruntime::test {
namespace {

struct Options {
  bool gemm{};
  bool reverse{};
  bool exposed_base{};
  bool exposed_low{};
  bool nonempty_default{};
  bool half{};
  bool different_provider{};
  bool broadcast_base{};
  float alpha{1.0f};
  const char* provider{kCpuExecutionProvider};
};

void BuildGraph(ModelTestBuilder& builder, const Options& opts,
                std::array<std::string, 4>& names) {
  constexpr int64_t K = 35, N = 16;
  const int64_t rank = opts.nonempty_default ? 3 : 0;
  const int32_t type = opts.half ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 : ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
  NodeArg* input = opts.half ? builder.MakeInput<MLFloat16>({2, K}, MLFloat16{1.0f}, MLFloat16{1.0f}) : builder.MakeInput<float>({2, K}, 1.0f, 1.0f);
  auto* packed = builder.MakeInitializer<uint8_t>({N, 2, 16}, uint8_t{0x88}, uint8_t{0x88});
  NodeArg* scales = opts.half ? builder.MakeInitializer<MLFloat16>({N, 2}, MLFloat16{1.0f}, MLFloat16{1.0f}) : builder.MakeInitializer<float>({N, 2}, 1.0f, 1.0f);
  auto* base = opts.exposed_base ? builder.MakeOutput() : builder.MakeIntermediate();
  auto& projection = builder.AddNode("MatMulNBits", {input, packed, scales}, {base}, kMSDomain);
  projection.AddAttribute("K", K);
  projection.AddAttribute("N", N);
  projection.AddAttribute("bits", int64_t{4});
  projection.AddAttribute("block_size", int64_t{32});
  projection.SetExecutionProviderType(opts.provider);
  if (opts.broadcast_base) {
    auto* row = builder.MakeScalarInitializer<int64_t>(0);
    auto* vector_base = builder.MakeIntermediate();
    auto& gather = builder.AddNode("Gather", {base, row}, {vector_base});
    gather.AddAttribute("axis", int64_t{0});
    gather.SetExecutionProviderType(opts.provider);
    base = vector_base;
  }
  auto* qa = builder.MakeInput<int8_t>({K, rank}, int8_t{0}, int8_t{0});
  auto* qb = builder.MakeInput<int8_t>({rank, N}, int8_t{0}, int8_t{0});
  auto* sa = builder.MakeInput<float>({2, rank}, 0.0f, 0.0f);
  auto* sb = builder.MakeInput<float>({(rank + 31) / 32, N}, 0.0f, 0.0f);
  names = {qa->Name(), qb->Name(), sa->Name(), sb->Name()};
  auto make_weight = [&](NodeArg* weights, NodeArg* scale) {
    auto* output = builder.MakeIntermediate();
    auto& dq = builder.AddNode("DequantizeLinear", {weights, scale}, {output});
    dq.AddAttribute("axis", int64_t{0});
    dq.AddAttribute("block_size", int64_t{32});
    dq.SetExecutionProviderType(opts.different_provider ? kCudaExecutionProvider : opts.provider);
    if (opts.half) {
      auto* cast_output = builder.MakeIntermediate();
      auto& cast = builder.AddNode("Cast", {output}, {cast_output});
      cast.AddAttribute("to", int64_t{type});
      cast.SetExecutionProviderType(opts.provider);
      return cast_output;
    }
    return output;
  };
  auto* a = make_weight(qa, sa);
  auto* b = make_weight(qb, sb);
  auto* low = opts.exposed_low ? builder.MakeOutput() : builder.MakeIntermediate();
  auto& first = builder.AddNode("MatMul", {input, a}, {low});
  first.SetExecutionProviderType(opts.provider);
  auto* output = builder.MakeOutput();
  if (opts.gemm) {
    auto& gemm = builder.AddNode("Gemm", {low, b, base}, {output});
    gemm.AddAttribute("alpha", opts.alpha);
    gemm.SetExecutionProviderType(opts.provider);
  } else {
    auto* delta = builder.MakeIntermediate();
    auto& second = builder.AddNode("MatMul", {low, b}, {delta});
    second.SetExecutionProviderType(opts.provider);
    auto& add = builder.AddNode("Add", opts.reverse ? std::vector<NodeArg*>{delta, base} : std::vector<NodeArg*>{base, delta}, {output});
    add.SetExecutionProviderType(opts.provider);
  }
}

void AddDefaults(Graph& graph, const std::array<std::string, 4>& names, int64_t rank) {
  const auto inputs = graph.GetInputsIncludingInitializers();
  graph.SetInputs(inputs);
  const std::array<std::array<int64_t, 2>, 4> shapes{{{35, rank}, {rank, 16}, {2, rank}, {(rank + 31) / 32, 16}}};
  for (size_t index = 0; index < names.size(); ++index) {
    ONNX_NAMESPACE::TensorProto tensor;
    tensor.set_name(names[index]);
    tensor.set_data_type(index < 2 ? ONNX_NAMESPACE::TensorProto_DataType_INT8 : ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    for (int64_t dim : shapes[index]) tensor.add_dims(dim);
    const size_t bytes = static_cast<size_t>(shapes[index][0] * shapes[index][1]) *
                         (index < 2 ? sizeof(int8_t) : sizeof(float));
    tensor.set_raw_data(std::string(bytes, '\0'));
    graph.AddInitializedTensor(tensor);
  }
}

void RunStructuralTest(const Options& options, bool expected) {
  std::array<std::string, 4> names;
  auto build = [&](ModelTestBuilder& builder) { BuildGraph(builder, options, names); };
  auto before = [&](Graph& graph) {
    AddDefaults(graph, names, options.nonempty_default ? 3 : 0);
    return graph.Resolve();
  };
  auto after = [&](Graph& graph) {
    const auto counts = CountOpsInGraph(graph);
    EXPECT_EQ(OpCount(counts, "com.microsoft.LoraMulAdd"), expected ? 1 : 0);
    EXPECT_EQ(OpCount(counts, "com.microsoft.MatMulNBits"), 1);
    if (expected) {
      EXPECT_EQ(OpCount(counts, "DequantizeLinear"), 0);
      EXPECT_EQ(OpCount(counts, "MatMul"), 0);
      for (const auto& name : names) {
        const auto& inputs = graph.GetInputsIncludingInitializers();
        EXPECT_TRUE(std::any_of(inputs.begin(), inputs.end(),
                                [&](const NodeArg* arg) { return arg->Name() == name; }));
      }
    }
    return Status::OK();
  };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, 21, DefaultLoggingManager().DefaultLogger(),
      std::make_unique<LoraMulAddFusion>(
          InlinedHashSet<std::string_view>{kCpuExecutionProvider, kWebGpuExecutionProvider}),
      TransformerLevel::Level2, 1, before, after));
}

}  // namespace

TEST(LoraMulAddFusion, QuantizedPatternsPreserveBaseAndInputs) {
  RunStructuralTest({}, true);
  Options options;
  options.reverse = true;
  RunStructuralTest(options, true);
  options.gemm = true;
  RunStructuralTest(options, true);
  options.alpha = 2.0f;
  RunStructuralTest(options, false);
}

TEST(LoraMulAddFusion, RejectsSharedValuesAndNonemptyDefaults) {
  Options options;
  options.exposed_base = true;
  RunStructuralTest(options, false);
  options = {};
  options.exposed_low = true;
  RunStructuralTest(options, false);
  options = {};
  options.nonempty_default = true;
  RunStructuralTest(options, false);
  options = {};
  options.different_provider = true;
  RunStructuralTest(options, false);
  options = {};
  options.broadcast_base = true;
  RunStructuralTest(options, false);
}

TEST(LoraMulAddFusion, HalfRequiresWebGpu) {
  Options options;
  options.half = true;
  RunStructuralTest(options, false);
  options.provider = kWebGpuExecutionProvider;
  RunStructuralTest(options, true);
}

TEST(LoraMulAddFusion, ExplicitOptInAndLegacyLocalAlias) {
  const auto provider = DefaultCpuExecutionProvider();
  ASSERT_NE(provider, nullptr);
  SessionOptions options;
  auto present = [&]() {
    const auto transformers = optimizer_utils::GenerateTransformers(
        TransformerLevel::Level2, options, *provider, DefaultLoggingManager().DefaultLogger());
    return std::any_of(transformers.begin(), transformers.end(), [](const auto& transformer) {
      return transformer->Name() == "LoraMulAddFusion";
    });
  };
  EXPECT_FALSE(present());
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("optimization.enable_matmul_nbits_lora_fusion", "1"));
  EXPECT_TRUE(present());
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("optimization.enable_lora_mul_add_fusion", "0"));
  EXPECT_FALSE(present());
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("optimization.enable_lora_mul_add_fusion", "1"));
  EXPECT_TRUE(present());
}

TEST(LoraMulAddFusion, QuantizedSessionDefaultActiveDefault) {
  Model model("quantized_lora_session", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 21}, {kMSDomain, 1}},
              {}, DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);
  std::array<std::string, 4> names;
  BuildGraph(builder, {}, names);
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());
  AddDefaults(graph, names, 0);
  const std::array<std::array<int64_t, 2>, 4> shapes{{{35, 0}, {0, 16}, {2, 0}, {0, 16}}};
  for (size_t index = 0; index < names.size(); ++index) {
    ONNX_NAMESPACE::TensorShapeProto shape;
    for (size_t axis = 0; axis < 2; ++axis) {
      auto* dim = shape.add_dim();
      if (shapes[index][axis] == 0) {
        dim->set_dim_param(index == 3 ? "rank_groups" : "rank");
      } else {
        dim->set_dim_value(shapes[index][axis]);
      }
    }
    graph.GetOrCreateNodeArg(names[index], nullptr).SetShape(shape);
    builder.feeds_.erase(names[index]);
  }
  ASSERT_STATUS_OK(graph.Resolve());
  const std::string bytes = model.ToProto().SerializeAsString();
  SessionOptions options;
  options.intra_op_param.thread_pool_size = 1;
  options.graph_optimization_level = TransformerLevel::Level2;
  InferenceSessionWrapper reference{options, GetEnvironment()};
  ASSERT_STATUS_OK(reference.Load(bytes.data(), static_cast<int>(bytes.size())));
  ASSERT_STATUS_OK(reference.Initialize());
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("optimization.enable_lora_mul_add_fusion", "1"));
  InferenceSessionWrapper fused{options, GetEnvironment()};
  ASSERT_STATUS_OK(fused.Load(bytes.data(), static_cast<int>(bytes.size())));
  ASSERT_STATUS_OK(fused.Initialize());
  ASSERT_EQ(OpCount(CountOpsInGraph(fused.GetGraph()), "com.microsoft.LoraMulAdd"), 1);
  EXPECT_EQ(OpCount(CountOpsInGraph(fused.GetGraph()), "com.microsoft.MatMulNBits"), 1);

  const auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  NameMLValMap active = builder.feeds_;
  CreateMLValue<int8_t>(allocator, {35, 3}, std::vector<int8_t>(105, 1), &active[names[0]]);
  CreateMLValue<int8_t>(allocator, {3, 16}, std::vector<int8_t>(48, 1), &active[names[1]]);
  CreateMLValue<float>(allocator, {2, 3}, {0.25f, 0.25f, 0.25f, 0.5f, 0.5f, 0.5f}, &active[names[2]]);
  CreateMLValue<float>(allocator, {1, 16}, std::vector<float>(16, 0.5f), &active[names[3]]);
  auto check = [&](const NameMLValMap& feeds, float expected) {
    for (auto* session : {&reference, &fused}) {
      std::vector<OrtValue> outputs;
      ASSERT_STATUS_OK(session->Run(RunOptions{}, feeds, builder.output_names_, &outputs));
      ASSERT_EQ(outputs.size(), 1u);
      const auto& output = outputs[0].Get<Tensor>();
      ASSERT_EQ(output.Shape(), TensorShape({2, 16}));
      for (float value : output.DataAsSpan<float>()) EXPECT_EQ(value, expected);
    }
  };
  check(builder.feeds_, 0.0f);
  check(active, 14.25f);
  check(builder.feeds_, 0.0f);
  OrtValue wrong_rank;
  CreateMLValue<int8_t>(allocator, {35, 4}, std::vector<int8_t>(140, 1), &wrong_rank);
  active[names[0]] = wrong_rank;
  std::vector<OrtValue> outputs;
  EXPECT_FALSE(fused.Run(RunOptions{}, active, builder.output_names_, &outputs).IsOK());
  check(builder.feeds_, 0.0f);
}

}  // namespace onnxruntime::test
#endif
