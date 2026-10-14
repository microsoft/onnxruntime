// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/matmul_nbits_qlora_fusion.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/optimizer/webgpu_fusion_test_util.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
#ifndef DISABLE_CONTRIB_OPS
namespace {

void BuildQLoRA(ModelTestBuilder& builder, int64_t m = 1, int64_t k = 2048, int64_t n = 128, int64_t batch = 1,
                bool expose = false, bool different_x = false, bool zero_base = false, int64_t adapter_accuracy = 0) {
  auto* x = builder.MakeInput<MLFloat16>({batch, m, k}, MLFloat16(-0.5f), MLFloat16(0.5f));
  auto* w = builder.MakeInitializer<uint8_t>({n, k / 32, 16}, zero_base ? uint8_t{0x88} : uint8_t{0},
                                             zero_base ? uint8_t{0x88} : uint8_t{255});
  auto* s = builder.MakeInitializer<MLFloat16>({n, k / 32}, MLFloat16(0.03125f), MLFloat16(0.0625f));
  auto* aq = builder.MakeInput<uint8_t>({32, k / 32, 32}, uint8_t{100}, uint8_t{156});
  auto* as = builder.MakeInput<MLFloat16>({32, k / 32}, MLFloat16(0.001f), MLFloat16(0.002f));
  auto* bq = builder.MakeInput<uint8_t>({n, 1, 32}, uint8_t{100}, uint8_t{156});
  auto* bs = builder.MakeInput<MLFloat16>({n, 1}, MLFloat16(0.001f), MLFloat16(0.002f));
  for (auto* arg : {aq, as, bq, bs}) {
    ONNX_NAMESPACE::TensorProto init;
    init.set_name(arg->Name());
    const bool packed = arg == aq || arg == bq;
    init.set_data_type(packed ? ONNX_NAMESPACE::TensorProto::UINT8 : ONNX_NAMESPACE::TensorProto::FLOAT16);
    int64_t count = 1;
    for (const auto& dim : arg->Shape()->dim()) {
      init.add_dims(dim.dim_value());
      count *= dim.dim_value();
    }
    if (packed) {
      init.set_raw_data(std::string(static_cast<size_t>(count), static_cast<char>(128)));
    } else {
      const std::vector<MLFloat16> ones(static_cast<size_t>(count), MLFloat16(1.0f));
      init.set_raw_data(ones.data(), ones.size() * sizeof(MLFloat16));
    }
    builder.graph_.AddInitializedTensor(init);
  }
  auto* base = builder.MakeIntermediate<MLFloat16>(std::vector<int64_t>{batch, m, n});
  auto* t = expose ? builder.MakeOutput<MLFloat16>(std::vector<int64_t>{batch, m, 32})
                   : builder.MakeIntermediate<MLFloat16>(std::vector<int64_t>{batch, m, 32});
  auto* delta = builder.MakeIntermediate<MLFloat16>(std::vector<int64_t>{batch, m, n});
  auto* y = builder.MakeOutput<MLFloat16>(std::vector<int64_t>{batch, m, n});
  auto* adapter_x = different_x ? builder.MakeInput<MLFloat16>({batch, m, k}, MLFloat16(-0.5f), MLFloat16(0.5f)) : x;
  if (different_x) {
    builder.graph_.SetInputs({x, aq, as, bq, bs, adapter_x});
  } else {
    builder.graph_.SetInputs({x, aq, as, bq, bs});
  }
  auto& base_node = builder.AddNode("MatMulNBits", {x, w, s}, {base}, kMSDomain);
  auto& down = builder.AddNode("MatMulNBits", {adapter_x, aq, as}, {t}, kMSDomain);
  auto& up = builder.AddNode("MatMulNBits", {t, bq, bs}, {delta}, kMSDomain);
  for (auto* node : {&base_node, &down, &up}) {
    node->AddAttribute("K", node == &up ? int64_t{32} : k);
    node->AddAttribute("N", node == &down ? int64_t{32} : n);
    node->AddAttribute("bits", node == &base_node ? int64_t{4} : int64_t{8});
    node->AddAttribute("block_size", int64_t{32});
    node->AddAttribute("accuracy_level", node == &base_node ? int64_t{4} : adapter_accuracy);
    node->SetExecutionProviderType(kWebGpuExecutionProvider);
  }
  builder.AddNode("Add", {base, delta}, {y}).SetExecutionProviderType(kWebGpuExecutionProvider);
}

Status CheckQLoRACount(const Graph& graph, size_t expected) {
  size_t count = 0;
  for (const auto& node : graph.Nodes()) count += node.OpType() == "MatMulNBitsQLoRA";
  ORT_RETURN_IF_NOT(count == expected, "Unexpected quantized LoRA fusion count.");
  return Status::OK();
}

}  // namespace

TEST_F(GraphTransformationTests, MatMulNBitsQLoRAFusionPreservesDefaults) {
  ASSERT_STATUS_OK(TestGraphTransformer(
      [](ModelTestBuilder& b) { BuildQLoRA(b); }, 21, *logger_, std::make_unique<MatMulNBitsQLoRAFusion>(),
      TransformerLevel::Level2, 1, nullptr,
      [](const Graph& graph) {
        ORT_RETURN_IF_ERROR(CheckQLoRACount(graph, 1));
        const auto inputs = graph.GetOverridableInitializers();
        ORT_RETURN_IF_NOT(inputs.size() == 4, "Adapter parameters were removed.");
        for (const auto* input : inputs) {
          const ONNX_NAMESPACE::TensorProto* init = nullptr;
          ORT_RETURN_IF_NOT(graph.GetInitializedTensor(input->Name(), init), "Missing default.");
          if (init->data_type() == ONNX_NAMESPACE::TensorProto::UINT8) {
            ORT_RETURN_IF_NOT(init->dims(0) == 0 || init->dims(1) == 0, "Packed default was not emptied.");
          } else {
            ORT_RETURN_IF_NOT(init->dims(0) > 0 && init->dims(1) > 0, "Scale default was emptied.");
          }
        }
        return Status::OK();
      }));
}

TEST_F(GraphTransformationTests, MatMulNBitsQLoRAFusionNumerics) {
  if (!DefaultWebGpuExecutionProvider()) GTEST_SKIP() << "WebGPU unavailable.";
  struct ShapeCase {
    int64_t m, k, n, batch;
    bool zero_base;
  };
  const ShapeCase shapes[] = {{1, 64, 128, 1, false}, {1, 2048, 128, 1, false}, {1, 3072, 132, 1, true}, {1, 8192, 128, 1, true}, {2, 2048, 132, 1, true}, {37, 2048, 128, 1, false}, {37, 3072, 132, 1, true}, {1, 2048, 128, 2, true}, {37, 2048, 132, 2, true}};
  for (const char* mode : {"", "auto", "fuse_down", "fuse_up"}) {
    for (const auto& shape : shapes) {
      SCOPED_TRACE(mode[0] ? mode : "default");
      SCOPED_TRACE(shape.m);
      SCOPED_TRACE(shape.k);
      SCOPED_TRACE(shape.n);
      SCOPED_TRACE(shape.batch);
      // DP4A prefill quantizes X; subgroup decode changes the K reduction
      // order. Keep exact comparisons for the other dispatch paths.
      const bool approximate =
          (shape.m >= 4 && std::string_view{mode} != "fuse_down") ||
          (shape.m == 1 && shape.batch == 1 && shape.k >= 2048 && shape.k <= 8192 &&
           std::string_view{mode} != "fuse_up");
      RunWebGpuFusionTransformerTest(
          [shape](ModelTestBuilder& b) {
            BuildQLoRA(b, shape.m, shape.k, shape.n, shape.batch, false, false, shape.zero_base);
          },
          [](InferenceSessionWrapper& s) { ASSERT_STATUS_OK(CheckQLoRACount(s.GetGraph(), 1)); },
          TransformerLevel::Level1, TransformerLevel::Level2, 21,
          approximate ? 0.002 : 0.0, approximate ? 0.001 : 0.0, std::make_unique<MatMulNBitsQLoRAFusion>(),
          [] { return DefaultWebGpuExecutionProvider(); },
          [mode](SessionOptions& options) {
            if (mode[0])
              ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.webgpu.lora_fusion_mode", mode));
          });
    }
  }
}

TEST_F(GraphTransformationTests, MatMulNBitsQLoRAFusionRejectsObservableIntermediate) {
  ASSERT_STATUS_OK(TestGraphTransformer([](ModelTestBuilder& b) { BuildQLoRA(b, 1, 2048, 128, 1, true); }, 21, *logger_,
                                        std::make_unique<MatMulNBitsQLoRAFusion>(), TransformerLevel::Level2, 1,
                                        nullptr, [](const Graph& g) { return CheckQLoRACount(g, 0); }));
}

TEST_F(GraphTransformationTests, MatMulNBitsQLoRAFusionFp32Accumulation) {
  if (!DefaultWebGpuExecutionProvider()) GTEST_SKIP() << "WebGPU unavailable.";
  for (int64_t m : {1, 2, 37}) {
    SCOPED_TRACE(m);
    const bool approximate = m == 1 || m >= 4;
    RunWebGpuFusionTransformerTest(
        [m](ModelTestBuilder& b) { BuildQLoRA(b, m, 2048, 132, 1, false, false, true); },
        [](InferenceSessionWrapper& s) { ASSERT_STATUS_OK(CheckQLoRACount(s.GetGraph(), 1)); },
        TransformerLevel::Level1, TransformerLevel::Level2, 21,
        approximate ? 0.002 : 0.0, approximate ? 0.001 : 0.0, std::make_unique<MatMulNBitsQLoRAFusion>(),
        [] {
          ConfigOptions options;
          ORT_THROW_IF_ERROR(options.AddConfigEntry(webgpu::options::kEnableMatmulFp32Accumulation,
                                                    webgpu::options::kEnableMatmulFp32Accumulation_ON));
          return WebGpuExecutionProviderWithOptions(options);
        });
  }
}

TEST_F(GraphTransformationTests, MatMulNBitsQLoRAFusionKeepsNonzeroOrSharedDefaults) {
  for (bool shared : {false, true}) {
    ASSERT_STATUS_OK(TestGraphTransformer(
        [shared](ModelTestBuilder& b) {
          BuildQLoRA(b);
          const NodeArg* input = nullptr;
          for (const auto& [name, tensor] : b.graph_.GetAllInitializedTensors()) {
            if (tensor->data_type() == ONNX_NAMESPACE::TensorProto::UINT8 && tensor->dims(0) == 32) {
              input = b.graph_.GetNodeArg(name);
              break;
            }
          }
          ORT_ENFORCE(input != nullptr);
          if (shared) {
            auto* observable = b.MakeOutput<uint8_t>(std::vector<int64_t>{32, 64, 32});
            b.AddNode("Identity", {const_cast<NodeArg*>(input)}, {observable});
          } else {
            const ONNX_NAMESPACE::TensorProto* original = nullptr;
            ORT_ENFORCE(b.graph_.GetInitializedTensor(input->Name(), original));
            auto replacement = *original;
            replacement.mutable_raw_data()->at(0) = static_cast<char>(129);
            b.graph_.RemoveInitializedTensor(input->Name());
            b.graph_.AddInitializedTensor(replacement);
          }
        },
        21, *logger_, std::make_unique<MatMulNBitsQLoRAFusion>(), TransformerLevel::Level2, 1, nullptr,
        [](const Graph& graph) {
          ORT_RETURN_IF_ERROR(CheckQLoRACount(graph, 1));
          for (const auto* input : graph.GetOverridableInitializers()) {
            const ONNX_NAMESPACE::TensorProto* init = nullptr;
            ORT_RETURN_IF_NOT(graph.GetInitializedTensor(input->Name(), init), "Missing default.");
            for (const auto dim : init->dims())
              ORT_RETURN_IF_NOT(dim > 0, "Observable or nonzero default was emptied.");
          }
          return Status::OK();
        }));
  }
}

TEST_F(GraphTransformationTests, MatMulNBitsQLoRAFusionRejectsDifferentInput) {
  ASSERT_STATUS_OK(TestGraphTransformer([](ModelTestBuilder& b) { BuildQLoRA(b, 1, 2048, 128, 1, false, true); }, 21,
                                        *logger_, std::make_unique<MatMulNBitsQLoRAFusion>(), TransformerLevel::Level2,
                                        1, nullptr, [](const Graph& g) { return CheckQLoRACount(g, 0); }));
}

TEST_F(GraphTransformationTests, MatMulNBitsQLoRAFusionRejectsDifferentAdapterAccuracy) {
  ASSERT_STATUS_OK(
      TestGraphTransformer([](ModelTestBuilder& b) { BuildQLoRA(b, 1, 2048, 128, 1, false, false, false, 4); }, 21,
                           *logger_, std::make_unique<MatMulNBitsQLoRAFusion>(), TransformerLevel::Level2, 1, nullptr,
                           [](const Graph& g) { return CheckQLoRACount(g, 0); }));
}

TEST_F(GraphTransformationTests, MatMulNBitsQLoRAFusionPartialOverrides) {
  if (!DefaultWebGpuExecutionProvider()) GTEST_SKIP() << "WebGPU unavailable.";
  Model model("quantized LoRA switching", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 21}, {kMSDomain, 1}}, {}, *logger_);
  auto& graph = model.MainGraph();
  ModelTestBuilder builder(graph);
  BuildQLoRA(builder, 1, 64);
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());
  const auto adapters = graph.GetOverridableInitializers();
  ASSERT_EQ(adapters.size(), 4u);
  std::string data;
  model.ToProto().SerializeToString(&data);
  std::vector<OrtValue> expected;
  for (bool fused : {false, true}) {
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Level1;
    InferenceSessionWrapper session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultWebGpuExecutionProvider()));
    ASSERT_STATUS_OK(session.Load(data.data(), static_cast<int>(data.size())));
    if (fused)
      ASSERT_STATUS_OK(
          session.RegisterGraphTransformer(std::make_unique<MatMulNBitsQLoRAFusion>(), TransformerLevel::Level2));
    ASSERT_STATUS_OK(session.Initialize());
    // Off, packed weights only, scales only, all parameters, then off again.
    for (int state = 0; state < 5; ++state) {
      auto feeds = builder.feeds_;
      for (const auto* input : adapters) {
        const bool packed = input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto::UINT8;
        if (state == 0 || state == 4 || (state == 1 && !packed) || (state == 2 && packed)) feeds.erase(input->Name());
      }
      std::vector<OrtValue> output;
      ASSERT_STATUS_OK(session.Run(RunOptions{}, feeds, builder.output_names_, &output));
      ASSERT_EQ(output.size(), 1u);
      if (!fused) {
        expected.push_back(std::move(output[0]));
      } else {
        const auto comparison = CompareOrtValue(output[0], expected[state], 0.0, 0.0, false);
        EXPECT_EQ(comparison.first, COMPARE_RESULT::SUCCESS) << "state=" << state << " " << comparison.second;
      }
    }
  }
}

#endif
}  // namespace onnxruntime::test
