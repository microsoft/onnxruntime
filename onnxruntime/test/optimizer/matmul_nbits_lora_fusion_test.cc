// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/matmul_nbits_fusion.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/test_environment.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"

#if !defined(DISABLE_CONTRIB_OPS) && !defined(ORT_MINIMAL_BUILD)
namespace onnxruntime::test {
namespace {

struct LoraFusionOptions {
  bool gemm{false};
  bool reverse_add{false};
  bool optional{true};
  bool expose_base{false};
  bool share_lora_a{false};
  bool different_provider{false};
  int64_t default_rank{0};
  float gemm_alpha{1.0f};
  const char* provider{kCpuExecutionProvider};
};

template <typename T>
void RunFusionTest(const LoraFusionOptions& options, bool expect_fusion) {
  constexpr int64_t K = 32;
  constexpr int64_t N = 16;
  std::string a_name;
  std::string b_name;
  auto build = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<T>({2, K}, T{1.0f}, T{1.0f});
    auto* packed = builder.MakeInitializer<uint8_t>({N, 1, 16}, uint8_t{0x88}, uint8_t{0x88});
    auto* scales = builder.MakeInitializer<T>({N, 1}, T{1.0f}, T{1.0f});
    auto* lora_a = builder.MakeInput<T>({K, options.default_rank}, T{0.0f}, T{0.0f});
    auto* lora_b = builder.MakeInput<T>({options.default_rank, N}, T{0.0f}, T{0.0f});
    a_name = lora_a->Name();
    b_name = lora_b->Name();
    auto* base_output = options.expose_base ? builder.MakeOutput() : builder.MakeIntermediate();
    auto* low_rank = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();
    auto& base = builder.AddNode("MatMulNBits", {input, packed, scales}, {base_output}, kMSDomain);
    base.AddAttribute("K", K);
    base.AddAttribute("N", N);
    base.AddAttribute("bits", int64_t{4});
    base.AddAttribute("block_size", int64_t{32});
    base.SetExecutionProviderType(options.provider);
    auto& first = builder.AddNode("MatMul", {input, lora_a}, {low_rank});
    first.SetExecutionProviderType(options.different_provider ? kCudaExecutionProvider : options.provider);
    if (options.share_lora_a) {
      builder.AddNode("Identity", {low_rank}, {builder.MakeOutput()});
    }
    if (options.gemm) {
      auto& gemm = builder.AddNode("Gemm", {low_rank, lora_b, base_output}, {output});
      gemm.AddAttribute("alpha", options.gemm_alpha);
      gemm.SetExecutionProviderType(options.provider);
    } else {
      auto* delta = builder.MakeIntermediate();
      auto& second = builder.AddNode("MatMul", {low_rank, lora_b}, {delta});
      second.SetExecutionProviderType(options.provider);
      auto& add = builder.AddNode("Add",
                                  options.reverse_add ? std::vector<NodeArg*>{delta, base_output}
                                                      : std::vector<NodeArg*>{base_output, delta},
                                  {output});
      add.SetExecutionProviderType(options.provider);
    }
  };
  auto before = [&](Graph& graph) {
    std::vector<const NodeArg*> inputs;
    for (const auto* input : graph.GetInputsIncludingInitializers()) {
      if (options.optional || (input->Name() != a_name && input->Name() != b_name)) {
        inputs.push_back(input);
      }
    }
    graph.SetInputs(inputs);
    for (const auto& [name, shape] :
         {std::pair{a_name, std::array<int64_t, 2>{K, options.default_rank}},
          std::pair{b_name, std::array<int64_t, 2>{options.default_rank, N}}}) {
      ONNX_NAMESPACE::TensorProto tensor;
      tensor.set_name(name);
      tensor.set_data_type(utils::ToTensorProtoElementType<T>());
      for (int64_t dimension : shape) {
        tensor.add_dims(dimension);
      }
      tensor.set_raw_data(std::string(static_cast<size_t>(shape[0] * shape[1]) * sizeof(T), '\0'));
      graph.AddInitializedTensor(tensor);
    }
    return graph.Resolve();
  };
  auto after = [&](Graph& graph) {
    const auto counts = CountOpsInGraph(graph);
    const auto found = counts.find("com.microsoft.MatMulNBitsLora");
    const int fused_count = found == counts.end() ? 0 : found->second;
    EXPECT_EQ(fused_count, expect_fusion ? 1 : 0);
    if (expect_fusion) {
      for (const auto& name : {a_name, b_name}) {
        const auto& inputs = graph.GetInputsIncludingInitializers();
        EXPECT_TRUE(std::any_of(inputs.begin(), inputs.end(), [&](const NodeArg* input) {
          return input->Name() == name;
        }));
        const auto* tensor = graph.GetInitializer(name, true);
        EXPECT_NE(tensor, nullptr);
        if (tensor) {
          EXPECT_TRUE(tensor->raw_data().empty());
        }
      }
    }
    return Status::OK();
  };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, 13, DefaultLoggingManager().DefaultLogger(),
      std::make_unique<MatMulNBitsLoraFusion>(
          InlinedHashSet<std::string_view>{kCpuExecutionProvider, kWebGpuExecutionProvider}),
      TransformerLevel::Level2, 1, before, after));
}

}  // namespace

TEST(MatMulNBitsLoraFusion, MatMulAddAndGemm) {
  RunFusionTest<float>({}, true);
  LoraFusionOptions options;
  options.reverse_add = true;
  RunFusionTest<float>(options, true);
  options.gemm = true;
  RunFusionTest<float>(options, true);
  options.gemm_alpha = 2.0f;
  RunFusionTest<float>(options, false);
}

TEST(MatMulNBitsLoraFusion, RejectUnsafeDefaultsAndIntermediates) {
  LoraFusionOptions options;
  options.optional = false;
  RunFusionTest<float>(options, false);
  options = {};
  options.default_rank = 2;
  RunFusionTest<float>(options, false);
  options = {};
  options.expose_base = true;
  RunFusionTest<float>(options, false);
  options = {};
  options.share_lora_a = true;
  RunFusionTest<float>(options, false);
  options = {};
  options.different_provider = true;
  RunFusionTest<float>(options, false);
}

TEST(MatMulNBitsLoraFusion, Float16RequiresWebGpu) {
  RunFusionTest<MLFloat16>({}, false);
  LoraFusionOptions options;
  options.provider = kWebGpuExecutionProvider;
  RunFusionTest<MLFloat16>(options, true);
}

TEST(MatMulNBitsLoraFusion, RequiresExplicitSessionOptIn) {
  const auto provider = DefaultCpuExecutionProvider();
  ASSERT_NE(provider, nullptr);
  SessionOptions options;
  auto contains_fusion = [&]() {
    const auto transformers = optimizer_utils::GenerateTransformers(
        TransformerLevel::Level2, options, *provider, DefaultLoggingManager().DefaultLogger());
    return std::any_of(transformers.begin(), transformers.end(), [](const auto& transformer) {
      return transformer->Name() == "MatMulNBitsLoraFusion";
    });
  };
  EXPECT_FALSE(contains_fusion());
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsEnableMatMulNBitsLoraFusion, "1"));
  EXPECT_TRUE(contains_fusion());
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsEnableMatMulNBitsLoraFusion, "0"));
  EXPECT_FALSE(contains_fusion());
}

TEST(MatMulNBitsLoraFusion, SymbolicRankDefaultActiveDefaultSession) {
  constexpr int64_t K = 32;
  constexpr int64_t N = 16;
  Model model("optional_lora_session", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 13}, {kMSDomain, 1}},
              {}, DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);
  auto* input = builder.MakeInput<float>({2, K}, 1.0f, 1.0f);
  auto* packed = builder.MakeInitializer<uint8_t>({N, 1, 16}, uint8_t{0x88}, uint8_t{0x88});
  auto* scales = builder.MakeInitializer<float>({N, 1}, 1.0f, 1.0f);
  auto make_slot = [&](const char* name, const std::array<int64_t, 2>& shape) {
    ONNX_NAMESPACE::TypeProto type;
    auto* tensor_type = type.mutable_tensor_type();
    tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    for (int64_t dimension : shape) {
      auto* dim = tensor_type->mutable_shape()->add_dim();
      if (dimension == 0) {
        dim->set_dim_param("lora_rank");
      } else {
        dim->set_dim_value(dimension);
      }
    }
    auto* slot = &graph.GetOrCreateNodeArg(name, &type);
    ONNX_NAMESPACE::TensorProto initializer;
    initializer.set_name(name);
    initializer.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    for (int64_t dimension : shape) {
      initializer.add_dims(dimension);
    }
    graph.AddInitializedTensor(initializer);
    return slot;
  };
  auto* lora_a = make_slot("lora_A", {K, 0});
  auto* lora_b = make_slot("lora_B", {0, N});
  auto* base_output = builder.MakeIntermediate();
  auto* low_rank = builder.MakeIntermediate();
  auto* delta = builder.MakeIntermediate();
  auto* output = builder.MakeOutput();
  auto& base = builder.AddNode("MatMulNBits", {input, packed, scales}, {base_output}, kMSDomain);
  base.AddAttribute("K", K);
  base.AddAttribute("N", N);
  base.AddAttribute("bits", int64_t{4});
  base.AddAttribute("block_size", int64_t{32});
  builder.AddNode("MatMul", {input, lora_a}, {low_rank});
  builder.AddNode("MatMul", {low_rank, lora_b}, {delta});
  builder.AddNode("Add", {base_output, delta}, {output});
  graph.SetInputs({input, lora_a, lora_b});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());
  const std::string bytes = model.ToProto().SerializeAsString();

  SessionOptions options;
  options.intra_op_param.thread_pool_size = 1;
  options.graph_optimization_level = TransformerLevel::Level2;
  InferenceSessionWrapper reference{options, GetEnvironment()};
  ASSERT_STATUS_OK(reference.Load(bytes.data(), static_cast<int>(bytes.size())));
  ASSERT_STATUS_OK(reference.Initialize());
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsEnableMatMulNBitsLoraFusion, "1"));
  InferenceSessionWrapper fused{options, GetEnvironment()};
  ASSERT_STATUS_OK(fused.Load(bytes.data(), static_cast<int>(bytes.size())));
  ASSERT_STATUS_OK(fused.Initialize());
  EXPECT_EQ(OpCount(CountOpsInGraph(reference.GetGraph()), "com.microsoft.MatMulNBitsLora"), 0);
  ASSERT_EQ(OpCount(CountOpsInGraph(fused.GetGraph()), "com.microsoft.MatMulNBitsLora"), 1);

  OrtValue active_a;
  OrtValue active_b;
  const auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  CreateMLValue<float>(allocator, {K, 2}, std::vector<float>(K * 2, 1.0f), &active_a);
  CreateMLValue<float>(allocator, {2, N}, std::vector<float>(2 * N, 1.0f), &active_b);
  NameMLValMap active = builder.feeds_;
  active.emplace("lora_A", active_a);
  active.emplace("lora_B", active_b);
  auto run_and_check = [&](const NameMLValMap& feeds, float expected) {
    for (auto* session : {&reference, &fused}) {
      std::vector<OrtValue> results;
      ASSERT_STATUS_OK(session->Run(RunOptions{}, feeds, builder.output_names_, &results));
      ASSERT_EQ(results.size(), 1u);
      const auto& tensor = results[0].Get<Tensor>();
      ASSERT_EQ(tensor.Shape(), TensorShape({2, N}));
      for (float value : tensor.DataAsSpan<float>()) {
        EXPECT_EQ(value, expected);
      }
    }
  };
  run_and_check(builder.feeds_, 0.0f);
  run_and_check(active, 64.0f);
  run_and_check(builder.feeds_, 0.0f);

  OrtValue wrong_rank;
  CreateMLValue<float>(allocator, {K, 3}, std::vector<float>(K * 3, 1.0f), &wrong_rank);
  active["lora_A"] = wrong_rank;
  std::vector<OrtValue> results;
  EXPECT_FALSE(fused.Run(RunOptions{}, active, builder.output_names_, &results).IsOK());
  run_and_check(builder.feeds_, 0.0f);
}

}  // namespace onnxruntime::test
#endif
