// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/node_attr_utils.h"
#include "core/optimizer/matmul_nbits_lora_fusion.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/optimizer/webgpu_fusion_test_util.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
#ifndef DISABLE_CONTRIB_OPS
namespace {

// Fused FP16 reductions can differ by a few ULPs across GPU architectures.
// Use two FP16 epsilons plus a small floor for values near cancellation.
constexpr double kFp16FusionAbsTolerance = 1e-4;
constexpr double kFp16FusionRelTolerance = 2.0 / 1024.0;

template <typename T = MLFloat16>
void BuildLoRA(ModelTestBuilder& builder, int64_t m = 1, bool zeros = false,
               bool expose_intermediate = false, bool different_x = false, int64_t rank = 32,
               bool flattened_weights = false, bool transposed = false, int64_t n = 128,
               int isolate = 0, int64_t k_override = 0, int64_t batch = 1) {
  const int64_t k = k_override ? k_override : transposed ? 2048
                                                         : 128;
  auto* x = builder.MakeInput<T>({batch, m, k}, T(-0.5f), T(0.5f));
  auto weight_shape = flattened_weights ? std::vector<int64_t>{n * k / 2} : std::vector<int64_t>{n, k / 32, 16};
  auto* w = builder.MakeInitializer<uint8_t>(weight_shape, isolate ? uint8_t{0x88} : uint8_t{0}, isolate ? uint8_t{0x89} : uint8_t{255});
  auto* scales = builder.MakeInitializer<T>({n, k / 32}, T(0.03125f), T(0.0625f));
  auto* a = builder.MakeInput<T>(transposed ? std::vector<int64_t>{rank, k} : std::vector<int64_t>{k, rank}, T(-0.0625f), T(0.0625f));
  auto* b = builder.MakeInput<T>(transposed ? std::vector<int64_t>{n, rank} : std::vector<int64_t>{rank, n}, T(-0.0625f), T(0.0625f));
  if (isolate == 1 || isolate == 2 || isolate == 4) {
    auto* arg = isolate == 2 ? a : b;
    auto values = builder.feeds_.at(arg->Name()).GetMutable<Tensor>()->MutableDataAsSpan<T>();
    std::fill(values.begin(), values.end(), T(0.0f));
    const int64_t cols = arg->Shape()->dim(1).dim_value();
    for (int64_t r = 0; r < rank; ++r) values[r * cols + r] = T(1.0f);
  }
  if (isolate == 4) {
    auto xv = builder.feeds_.at(x->Name()).GetMutable<Tensor>()->MutableDataAsSpan<T>();
    auto av = builder.feeds_.at(a->Name()).GetMutable<Tensor>()->MutableDataAsSpan<T>();
    std::fill(xv.begin(), xv.end(), T(0.0f));
    std::fill(av.begin(), av.end(), T(0.0f));
    xv[0] = T(10000.0f);
    xv[1] = T(0.001f);
    xv[2] = T(-10000.0f);
    xv[3] = T(0.001f);
    for (int i = 0; i < 4; ++i) av[i] = T(1.0f);
  }
  if (zeros) {
    for (auto* arg : {a, b}) {
      ONNX_NAMESPACE::TensorProto init;
      init.set_name(arg->Name());
      init.set_data_type(utils::ToTensorProtoElementType<T>());
      int64_t count = 1;
      for (const auto& dim : arg->Shape()->dim()) {
        init.add_dims(dim.dim_value());
        count *= dim.dim_value();
      }
      init.set_raw_data(std::string(static_cast<size_t>(count) * sizeof(T), '\0'));
      builder.graph_.AddInitializedTensor(init);
    }
    builder.graph_.SetInputs({x, a, b});
  }
  auto* base = builder.MakeIntermediate<T>(std::vector<int64_t>{batch, m, n});
  auto* t = expose_intermediate ? builder.MakeOutput<T>(std::vector<int64_t>{batch, m, rank})
                                : builder.MakeIntermediate<T>(std::vector<int64_t>{batch, m, rank});
  auto* delta = builder.MakeIntermediate<T>(std::vector<int64_t>{batch, m, n});
  auto* y = builder.MakeOutput<T>(std::vector<int64_t>{batch, m, n});
  NodeAttributes attrs;
  for (auto [name, value] : {std::pair{"K", k}, {"N", n}, {"bits", int64_t{4}}, {"block_size", int64_t{32}}}) {
    utils::SetNodeAttribute(utils::MakeAttribute(name, value), attrs);
  }
  auto& base_node = builder.AddNode("MatMulNBits", {x, w, scales}, {base}, kMSDomain);
  for (const auto& [name, attr] : attrs) base_node.AddAttributeProto(attr);
  base_node.AddAttribute("accuracy_level", int64_t{4});
  auto* lora_x = different_x ? builder.MakeInput<T>({batch, m, k}, T(0.0f), T(1.0f)) : x;
  auto* matmul_a = a;
  auto* matmul_b = b;
  if (transposed) {
    matmul_a = builder.MakeIntermediate<T>(std::vector<int64_t>{k, rank});
    matmul_b = builder.MakeIntermediate<T>(std::vector<int64_t>{rank, n});
    auto& ta = builder.AddNode("Transpose", {a}, {matmul_a});
    auto& tb = builder.AddNode("Transpose", {b}, {matmul_b});
    for (auto* node : {&ta, &tb}) {
      node->AddAttribute("perm", std::vector<int64_t>{1, 0});
      node->SetExecutionProviderType(kWebGpuExecutionProvider);
    }
  }
  auto& down = builder.AddNode("MatMul", {lora_x, matmul_a}, {t});
  auto& up = builder.AddNode("MatMul", {t, matmul_b}, {delta});
  auto& add = builder.AddNode("Add", {delta, base}, {y});
  for (auto* node : {&base_node, &down, &up, &add}) node->SetExecutionProviderType(kWebGpuExecutionProvider);
}

Status CheckCount(const Graph& graph, size_t count) {
  size_t found = 0;
  for (const auto& node : graph.Nodes()) found += node.OpType() == "MatMulNBitsLoRA";
  ORT_RETURN_IF_NOT(found == count, "Unexpected fused LoRA count: ", found);
  return Status::OK();
}

}  // namespace

TEST_F(GraphTransformationTests, MatMulNBitsLoRAFusionPreservesOverridableInputs) {
  ASSERT_STATUS_OK(TestGraphTransformer(
      [](ModelTestBuilder& b) { BuildLoRA(b, 1, true); }, 21, *logger_,
      std::make_unique<MatMulNBitsLoRAFusion>(), TransformerLevel::Level2, 1, nullptr,
      [](const Graph& g) {
        ORT_RETURN_IF_ERROR(CheckCount(g, 1));
        ORT_RETURN_IF_NOT(g.GetOverridableInitializers().size() == 2, "Adapter inputs were removed.");
        for (const auto* arg : g.GetOverridableInitializers()) {
          const ONNX_NAMESPACE::TensorProto* init = nullptr;
          ORT_RETURN_IF_NOT(g.GetInitializedTensor(arg->Name(), init), "Missing adapter default.");
          ORT_RETURN_IF_NOT(init->dims(0) == 0 || init->dims(1) == 0, "Default was not emptied.");
        }
        return Status::OK();
      }));
}

TEST_F(GraphTransformationTests, MatMulNBitsLoRAFusionTransposedAdapters) {
  ASSERT_STATUS_OK(TestGraphTransformer(
      [](ModelTestBuilder& b) { BuildLoRA(b, 1, true, false, false, 32, false, true); }, 21, *logger_,
      std::make_unique<MatMulNBitsLoRAFusion>(), TransformerLevel::Level2, 1, nullptr,
      [](const Graph& g) {
        ORT_RETURN_IF_ERROR(CheckCount(g, 1));
        for (const auto& node : g.Nodes()) {
          ORT_RETURN_IF_NOT(node.OpType() != "Transpose", "Adapter transpose was not absorbed.");
          if (node.OpType() == "MatMulNBitsLoRA") {
            ORT_RETURN_IF_NOT(node.GetAttributes().at("adapter_transposed").i() == 1, "Missing layout flag.");
          }
        }
        ORT_RETURN_IF_NOT(g.GetOverridableInitializers().size() == 2, "Adapter inputs were removed.");
        return Status::OK();
      }));
}

TEST_F(GraphTransformationTests, MatMulNBitsLoRAFusionTransposedNumerics) {
  if (!DefaultWebGpuExecutionProvider()) GTEST_SKIP() << "WebGPU unavailable.";
  for (const char* mode : {"", "auto", "fuse_down", "fuse_up"}) {
    SCOPED_TRACE(mode[0] ? mode : "default");
    for (int64_t m : {1, 2, 37}) {
      for (int64_t n : {128, 132}) {
        SCOPED_TRACE(m);
        SCOPED_TRACE(n);
        RunWebGpuFusionTransformerTest(
            [m, n](ModelTestBuilder& b) { BuildLoRA(b, m, true, false, false, 32, false, true, n); },
            [](InferenceSessionWrapper& s) { ASSERT_STATUS_OK(CheckCount(s.GetGraph(), 1)); },
            TransformerLevel::Level1, TransformerLevel::Level2, 21,
            kFp16FusionAbsTolerance, kFp16FusionRelTolerance,
            std::make_unique<MatMulNBitsLoRAFusion>(), [] { return DefaultWebGpuExecutionProvider(); },
            [mode](SessionOptions& options) {
              if (mode[0]) {
                ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.webgpu.lora_fusion_mode", mode));
              }
            });
      }
    }
  }
}

TEST_F(GraphTransformationTests, MatMulNBitsLoRAFusionTransposedProjectionNumerics) {
  if (!DefaultWebGpuExecutionProvider()) GTEST_SKIP() << "WebGPU unavailable.";
  // An identity up/down adapter isolates each projection. Zero base weights
  // expose low-magnitude errors; the last case stresses FP32 cancellation.
  for (const char* mode : {"", "auto", "fuse_down", "fuse_up"}) {
    SCOPED_TRACE(mode[0] ? mode : "default");
    for (int isolate : {1, 2, 3, 4}) {
      for (int64_t k : {2048, 3072, 8192}) {
        SCOPED_TRACE(isolate);
        SCOPED_TRACE(k);
        RunWebGpuFusionTransformerTest(
            [isolate, k](ModelTestBuilder& b) { BuildLoRA(b, 1, true, false, false, 32, false, true, 2048, isolate, k); },
            [](InferenceSessionWrapper& s) { ASSERT_STATUS_OK(CheckCount(s.GetGraph(), 1)); },
            TransformerLevel::Level1, TransformerLevel::Level2, 21,
            isolate == 4 ? 0.0 : kFp16FusionAbsTolerance, isolate == 4 ? 0.0 : kFp16FusionRelTolerance,
            std::make_unique<MatMulNBitsLoRAFusion>(), [] { return DefaultWebGpuExecutionProvider(); },
            [mode](SessionOptions& options) {
              if (mode[0]) {
                ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.webgpu.lora_fusion_mode", mode));
              }
            });
      }
    }
  }
}

TEST_F(GraphTransformationTests, MatMulNBitsLoRAFusionTransposedBatchedNumerics) {
  if (!DefaultWebGpuExecutionProvider()) GTEST_SKIP() << "WebGPU unavailable.";
  for (const char* mode : {"", "auto", "fuse_down", "fuse_up"}) {
    SCOPED_TRACE(mode[0] ? mode : "default");
    for (int64_t batch : {2, 4}) {
      for (int64_t m : {1, 3, 37}) {
        SCOPED_TRACE(batch);
        SCOPED_TRACE(m);
        RunWebGpuFusionTransformerTest(
            [batch, m](ModelTestBuilder& b) { BuildLoRA(b, m, true, false, false, 32, false, true, 128, 0, 0, batch); },
            [](InferenceSessionWrapper& s) { ASSERT_STATUS_OK(CheckCount(s.GetGraph(), 1)); },
            TransformerLevel::Level1, TransformerLevel::Level2, 21,
            kFp16FusionAbsTolerance, kFp16FusionRelTolerance,
            std::make_unique<MatMulNBitsLoRAFusion>(), [] { return DefaultWebGpuExecutionProvider(); },
            [mode](SessionOptions& options) {
              if (mode[0]) {
                ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.webgpu.lora_fusion_mode", mode));
              }
            });
      }
    }
  }
}

TEST_F(GraphTransformationTests, MatMulNBitsLoRAFusionRejectsObservableIntermediate) {
  ASSERT_STATUS_OK(TestGraphTransformer(
      [](ModelTestBuilder& b) { BuildLoRA(b, 1, false, true); }, 21, *logger_,
      std::make_unique<MatMulNBitsLoRAFusion>(), TransformerLevel::Level2, 1, nullptr,
      [](const Graph& g) { return CheckCount(g, 0); }));
}

TEST_F(GraphTransformationTests, MatMulNBitsLoRAFusionRejectsDifferentActivation) {
  ASSERT_STATUS_OK(TestGraphTransformer(
      [](ModelTestBuilder& b) { BuildLoRA(b, 1, false, false, true); }, 21, *logger_,
      std::make_unique<MatMulNBitsLoRAFusion>(), TransformerLevel::Level2, 1, nullptr,
      [](const Graph& g) { return CheckCount(g, 0); }));
}

TEST_F(GraphTransformationTests, MatMulNBitsLoRAFusionRejectsUnsupportedRank) {
  ASSERT_STATUS_OK(TestGraphTransformer(
      [](ModelTestBuilder& b) { BuildLoRA(b, 1, false, false, false, 16); }, 21, *logger_,
      std::make_unique<MatMulNBitsLoRAFusion>(), TransformerLevel::Level2, 1, nullptr,
      [](const Graph& g) { return CheckCount(g, 0); }));
}

TEST_F(GraphTransformationTests, MatMulNBitsLoRAFusionLeavesLegacyPackedShapesAlone) {
  ASSERT_STATUS_OK(TestGraphTransformer(
      [](ModelTestBuilder& b) { BuildLoRA(b, 1, false, false, false, 32, true); }, 21, *logger_,
      std::make_unique<MatMulNBitsLoRAFusion>(), TransformerLevel::Level2, 1, nullptr,
      [](const Graph& g) { return CheckCount(g, 0); }));
}

TEST_F(GraphTransformationTests, MatMulNBitsLoRAFusionMatchesWebGpuDecodeAndPrefill) {
  if (!DefaultWebGpuExecutionProvider()) GTEST_SKIP() << "WebGPU unavailable.";
  for (int64_t m : {1, 37}) {
    SCOPED_TRACE(m);
    RunWebGpuFusionTransformerTest(
        [m](ModelTestBuilder& b) { BuildLoRA(b, m, true); },
        [](InferenceSessionWrapper& s) { ASSERT_STATUS_OK(CheckCount(s.GetGraph(), 1)); },
        TransformerLevel::Level1, TransformerLevel::Level2, 21,
        kFp16FusionAbsTolerance, kFp16FusionRelTolerance,
        std::make_unique<MatMulNBitsLoRAFusion>(), [] { return DefaultWebGpuExecutionProvider(); });
  }
}

template <typename T>
void RunLoRASwitchesFeedsAndDefaults(const logging::Logger& logger) {
  if (!DefaultWebGpuExecutionProvider()) GTEST_SKIP() << "WebGPU unavailable.";
  Model model("LoRA switching", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 21}, {kMSDomain, 1}}, {}, logger);
  auto& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  BuildLoRA<T>(helper, 1, true);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());
  const auto adapters = graph.GetOverridableInitializers();
  ASSERT_EQ(adapters.size(), 2u);
  std::string data;
  model.ToProto().SerializeToString(&data);
  std::vector<OrtValue> expected;
  for (bool fused : {false, true}) {
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Level1;
    InferenceSessionWrapper session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultWebGpuExecutionProvider()));
    ASSERT_STATUS_OK(session.Load(data.data(), static_cast<int>(data.size())));
    if (fused) {
      ASSERT_STATUS_OK(session.RegisterGraphTransformer(std::make_unique<MatMulNBitsLoRAFusion>(),
                                                        TransformerLevel::Level2));
    }
    ASSERT_STATUS_OK(session.Initialize());
    // Off -> adapter A -> adapter B -> partially supplied -> off.
    for (int state = 0; state < 5; ++state) {
      auto feeds = helper.feeds_;
      for (size_t i = 0; i < adapters.size(); ++i) {
        const auto& name = adapters[i]->Name();
        if (state == 0 || state == 4 || (state == 3 && i == 1)) {
          feeds.erase(name);
        } else {
          auto* tensor = feeds.at(name).GetMutable<Tensor>();
          auto values = tensor->MutableDataAsSpan<T>();
          std::fill(values.begin(), values.end(), T(state == 2 ? -0.03125f : 0.0625f));
        }
      }
      std::vector<OrtValue> outputs;
      ASSERT_STATUS_OK(session.Run(RunOptions{}, feeds, helper.output_names_, &outputs));
      ASSERT_EQ(outputs.size(), 1u);
      if (fused) {
        auto result = CompareOrtValue(outputs[0], expected[state], 0.0, 0.0, false);
        EXPECT_EQ(result.first, COMPARE_RESULT::SUCCESS) << "state=" << state << " " << result.second;
      } else {
        expected.push_back(std::move(outputs[0]));
      }
    }
  }
}
TEST_F(GraphTransformationTests, MatMulNBitsLoRAFusionSwitchesFeedsAndDefaults) {
  RunLoRASwitchesFeedsAndDefaults<MLFloat16>(*logger_);
}
TEST_F(GraphTransformationTests, MatMulNBitsLoRAFusionSwitchesFeedsAndDefaultsFp32) {
  RunLoRASwitchesFeedsAndDefaults<float>(*logger_);
}

TEST_F(GraphTransformationTests, MatMulNBitsLoRAFusionFp32Numerics) {
  if (!DefaultWebGpuExecutionProvider()) GTEST_SKIP() << "WebGPU unavailable.";
  for (bool transposed : {false, true}) {
    for (const char* mode : {"auto", "fuse_down", "fuse_up"}) {
      for (int64_t m : {1, 2, 37}) {
        for (int64_t n : {128, 132}) {
          for (int64_t batch : {1, 2}) {
            SCOPED_TRACE(MakeString("transposed=", transposed, ", mode=", mode, ", M=", m,
                                    ", N=", n, ", batch=", batch));
            RunWebGpuFusionTransformerTest(
                [=](ModelTestBuilder& b) { BuildLoRA<float>(b, m, true, false, false, 32, false, transposed, n, 0, 2048, batch); },
                [](InferenceSessionWrapper& s) { ASSERT_STATUS_OK(CheckCount(s.GetGraph(), 1)); },
                TransformerLevel::Level1, TransformerLevel::Level2, 21, 1e-5, 1e-5,
                std::make_unique<MatMulNBitsLoRAFusion>(), [] { return DefaultWebGpuExecutionProvider(); },
                [mode](SessionOptions& options) {
                  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.webgpu.lora_fusion_mode", mode));
                });
          }
        }
      }
    }
  }
}
#endif
}  // namespace onnxruntime::test
