// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/matmul_nbits_qlora_fusion.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/IOBinding.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/optimizer/webgpu_fusion_test_util.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
#ifndef DISABLE_CONTRIB_OPS
namespace {

template <typename T = MLFloat16, bool ExactBase = false>
void BuildQLoRA(ModelTestBuilder& builder, int64_t m = 1, int64_t k = 2048, int64_t n = 128, int64_t batch = 1,
                bool expose = false, bool different_x = false, bool zero_base = false, int64_t adapter_accuracy = 0) {
  auto* x = builder.MakeInput<T>({batch, m, k}, T(-0.5f), T(0.5f));
  // Integer random ranges exclude the upper bound, even for constant weights.
  const uint8_t base_weight_min = zero_base ? uint8_t{0x88} : (ExactBase ? uint8_t{0x99} : uint8_t{0});
  const uint8_t base_weight_max = zero_base ? uint8_t{0x89} : (ExactBase ? uint8_t{0x9a} : uint8_t{255});
  auto* w = builder.MakeInitializer<uint8_t>({n, k / 32, 16}, base_weight_min, base_weight_max);
  auto* s = builder.MakeInitializer<T>({n, k / 32}, T(ExactBase ? 0.0625f : 0.03125f), T(0.0625f));
  auto* aq = builder.MakeInput<uint8_t>({32, k / 32, 32}, uint8_t{100}, uint8_t{156});
  auto* as = builder.MakeInput<T>({32, k / 32}, T(0.001f), T(0.002f));
  auto* bq = builder.MakeInput<uint8_t>({n, 1, 32}, uint8_t{100}, uint8_t{156});
  auto* bs = builder.MakeInput<T>({n, 1}, T(0.001f), T(0.002f));
  for (auto* arg : {aq, as, bq, bs}) {
    ONNX_NAMESPACE::TensorProto init;
    init.set_name(arg->Name());
    const bool packed = arg == aq || arg == bq;
    init.set_data_type(packed ? ONNX_NAMESPACE::TensorProto::UINT8 : utils::ToTensorProtoElementType<T>());
    int64_t count = 1;
    for (const auto& dim : arg->Shape()->dim()) {
      init.add_dims(dim.dim_value());
      count *= dim.dim_value();
    }
    if (packed) {
      init.set_raw_data(std::string(static_cast<size_t>(count), static_cast<char>(128)));
    } else {
      const std::vector<T> ones(static_cast<size_t>(count), T(1.0f));
      init.set_raw_data(ones.data(), ones.size() * sizeof(T));
    }
    builder.graph_.AddInitializedTensor(init);
  }
  auto* base = builder.MakeIntermediate<T>(std::vector<int64_t>{batch, m, n});
  auto* t = expose ? builder.MakeOutput<T>(std::vector<int64_t>{batch, m, 32})
                   : builder.MakeIntermediate<T>(std::vector<int64_t>{batch, m, 32});
  auto* delta = builder.MakeIntermediate<T>(std::vector<int64_t>{batch, m, n});
  auto* y = builder.MakeOutput<T>(std::vector<int64_t>{batch, m, n});
  auto* adapter_x = different_x ? builder.MakeInput<T>({batch, m, k}, T(-0.5f), T(0.5f)) : x;
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
  const ShapeCase shapes[] = {
      {1, 64, 1, 1, false},
      {1, 2176, 127, 1, true},
      {2, 2048, 129, 2, true},
      {37, 2048, 128, 1, false},
      {37, 2176, 127, 2, true},
  };
  for (const auto& shape : shapes) {
    SCOPED_TRACE(shape.m);
    SCOPED_TRACE(shape.k);
    SCOPED_TRACE(shape.n);
    SCOPED_TRACE(shape.batch);
    // FP16 fusion can change rounding on Metal even without DP4A or subgroup
    // reduction. Use the FP16 error budget for every dispatch path.
    RunWebGpuFusionTransformerTest(
        [shape](ModelTestBuilder& b) {
          BuildQLoRA(b, shape.m, shape.k, shape.n, shape.batch, false, false, shape.zero_base);
        },
        [](InferenceSessionWrapper& s) { ASSERT_STATUS_OK(CheckQLoRACount(s.GetGraph(), 1)); },
        TransformerLevel::Level1, TransformerLevel::Level2, 21,
        0.002, 0.001, std::make_unique<MatMulNBitsQLoRAFusion>(),
        [] { return DefaultWebGpuExecutionProvider(); });
  }
}

TEST_F(GraphTransformationTests, MatMulNBitsQLoRAFusionFp32Numerics) {
  if (!DefaultWebGpuExecutionProvider()) GTEST_SKIP() << "WebGPU unavailable.";
  for (const auto& [m, n] : {std::pair<int64_t, int64_t>{1, 128}, {2, 128}, {37, 128}, {2, 129}}) {
    SCOPED_TRACE(MakeString("M=", m, ", N=", n));
    // Eligible FP32 decode and short prefill share quantized X with the base.
    const bool quantized_down = m >= 4 || n % 16 == 0;
    RunWebGpuFusionTransformerTest(
        [m, n](ModelTestBuilder& b) { BuildQLoRA<float>(b, m, 2048, n); },
        [](InferenceSessionWrapper& s) { ASSERT_STATUS_OK(CheckQLoRACount(s.GetGraph(), 1)); },
        TransformerLevel::Level1, TransformerLevel::Level2, 21,
        quantized_down ? 0.002 : 0.00001, quantized_down ? 0.001 : 0.00001,
        std::make_unique<MatMulNBitsQLoRAFusion>(), [] { return DefaultWebGpuExecutionProvider(); });
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
    // FP32 accumulation still produces FP16 intermediates and output, so
    // fused and unfused execution need not be bitwise identical.
    RunWebGpuFusionTransformerTest(
        [m](ModelTestBuilder& b) { BuildQLoRA(b, m, 2048, 132, 1, false, false, true); },
        [](InferenceSessionWrapper& s) { ASSERT_STATUS_OK(CheckQLoRACount(s.GetGraph(), 1)); },
        TransformerLevel::Level1, TransformerLevel::Level2, 21,
        0.002, 0.001, std::make_unique<MatMulNBitsQLoRAFusion>(),
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

template <typename T>
void RunQLoRASerializedEmptyDefaults(const logging::Logger& logger, int64_t k = 64) {
  if (!DefaultWebGpuExecutionProvider()) GTEST_SKIP() << "WebGPU unavailable.";
  for (int64_t m : {1, 2, 37}) {
    SCOPED_TRACE(m);
    Model model("quantized LoRA empty defaults", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 21}, {kMSDomain, 1}}, {}, logger);
    auto& graph = model.MainGraph();
    ModelTestBuilder builder(graph);
    // A random base can round differently when the subgroup kernel accumulates
    // adapter-up into it. Keep this state/capture test exact; Numerics covers rounding.
    BuildQLoRA<T, true>(builder, m, k);
    builder.SetGraphOutputs();
    ASSERT_STATUS_OK(graph.Resolve());
    const auto adapters = graph.GetOverridableInitializers();
    ASSERT_EQ(adapters.size(), 4u);
    std::string a_name, b_name;
    // Dyadic inputs keep the switching checks exact on all reduction paths.
    for (auto& [name, value] : builder.feeds_) {
      Tensor* tensor = value.template GetMutable<Tensor>();
      if (tensor->IsDataType<uint8_t>()) {
        auto bytes = tensor->MutableDataAsSpan<uint8_t>();
        std::fill(bytes.begin(), bytes.end(), uint8_t{129});
        (tensor->Shape()[0] == 32 ? a_name : b_name) = name;
      } else {
        auto values = tensor->MutableDataAsSpan<T>();
        const float fill = tensor->Shape().NumDimensions() == 3 ? 0.25f : 0.03125f;
        std::fill(values.begin(), values.end(), T(fill));
      }
    }
    ASSERT_FALSE(a_name.empty());
    ASSERT_FALSE(b_name.empty());
    std::vector<OrtValue> expected;
    // Compare unfused, load-time fusion, and serialized fusion using the same
    // partial-override states and exact reference values.
    for (int mode : {0, 1, 2}) {
      const bool fused = mode != 0;
      if (mode == 2) {
        bool modified = false;
        ASSERT_STATUS_OK(MatMulNBitsQLoRAFusion().Apply(graph, modified, logger));
        ASSERT_TRUE(modified);
        ASSERT_STATUS_OK(graph.Resolve());
      }
      const auto proto = model.ToProto();
      if (mode == 2) {
        ASSERT_EQ(proto.graph().node_size(), 1);
        ASSERT_EQ(proto.graph().node(0).op_type(), "MatMulNBitsQLoRA");
        int empty_count = 0;
        for (const auto& init : proto.graph().initializer()) {
          if (init.name() != a_name && init.name() != b_name) continue;
          ++empty_count;
          ASSERT_EQ(init.dims_size(), 3);
          EXPECT_EQ(init.dims(init.name() == a_name ? 0 : 1), 0);
          EXPECT_TRUE(init.raw_data().empty());
          EXPECT_EQ(init.int32_data_size(), 0);
          EXPECT_EQ(init.external_data_size(), 0);
        }
        ASSERT_EQ(empty_count, 2);
      }
      std::string data;
      ASSERT_TRUE(proto.SerializeToString(&data));
      SessionOptions options;
      options.graph_optimization_level = TransformerLevel::Level1;
      InferenceSessionWrapper session(options, GetEnvironment());
      ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultWebGpuExecutionProvider()));
      ASSERT_STATUS_OK(session.Load(data.data(), static_cast<int>(data.size())));
      if (mode == 1) {
        ASSERT_STATUS_OK(
            session.RegisterGraphTransformer(std::make_unique<MatMulNBitsQLoRAFusion>(), TransformerLevel::Level2));
      }
      // Mode 2 loads the serialized operator without a LoRA transformer.
      ASSERT_STATUS_OK(session.Initialize());
      ASSERT_STATUS_OK(CheckQLoRACount(session.GetGraph(), fused ? 1 : 0));
      // Off, A only, B only, scales only, packed weights only, all, off again.
      for (int state = 0; state < 7; ++state) {
        SCOPED_TRACE(state);
        auto feeds = builder.feeds_;
        for (const auto* input : adapters) {
          const auto& name = input->Name();
          const bool packed = name == a_name || name == b_name;
          const bool supply = (state == 1 && name == a_name) || (state == 2 && name == b_name) ||
                              (state == 3 && !packed) || (state == 4 && packed) || state == 5;
          if (!supply) feeds.erase(name);
        }
        std::vector<OrtValue> output;
        ASSERT_STATUS_OK(session.Run(RunOptions{}, feeds, builder.output_names_, &output));
        ASSERT_EQ(output.size(), 1u);
        // X=1/4 and base weights=1/16 give K/64. Unit adapter weights give
        // 8*K with default scales, or K/128 when both scales are 1/32.
        const float base_value = static_cast<float>(k) / 64.0f;
        const float delta = state == 4 ? 8.0f * static_cast<float>(k)
                                       : (state == 5 ? static_cast<float>(k) / 128.0f : 0.0f);
        const auto values = output[0].Get<Tensor>().DataAsSpan<T>();
        EXPECT_TRUE(std::all_of(values.begin(), values.end(),
                                [base_value, delta](T value) { return value == T(base_value + delta); }))
            << "Expected every output to equal " << base_value + delta;
        if (!fused) {
          expected.push_back(std::move(output[0]));
        } else {
          const auto comparison = CompareOrtValue(output[0], expected[state], 0.0, 0.0, false);
          EXPECT_EQ(comparison.first, COMPARE_RESULT::SUCCESS) << comparison.second;
        }
      }
    }
    EXPECT_NE(CompareOrtValue(expected[0], expected[5], 0.0, 0.0, false).first, COMPARE_RESULT::SUCCESS);

    ConfigOptions config;
    ASSERT_STATUS_OK(config.AddConfigEntry(webgpu::options::kEnableGraphCapture,
                                           webgpu::options::kEnableGraphCapture_ON));
    auto ep = WebGpuExecutionProviderWithOptions(config);
    ASSERT_NE(ep, nullptr);
    auto* capture_provider = ep.get();
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Level1;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
    InferenceSession session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(ep)));
    const auto data = model.ToProto().SerializeAsString();
    ASSERT_STATUS_OK(session.Load(data.data(), static_cast<int>(data.size())));
    ASSERT_STATUS_OK(session.Initialize());
    const auto allocators = capture_provider->CreatePreferredAllocators();
    ASSERT_FALSE(allocators.empty());
    const auto allocator = allocators.back();
    ASSERT_NE(allocator->Info().device.Type(), OrtDevice::CPU);
    std::unique_ptr<IOBinding> bindings[2];
    OrtValue outputs[2];
    for (int state : {0, 1}) {
      ASSERT_STATUS_OK(session.NewIOBinding(&bindings[state]));
      for (const auto& [name, value] : builder.feeds_) {
        const bool adapter = std::any_of(adapters.begin(), adapters.end(),
                                         [&name](const NodeArg* arg) { return arg->Name() == name; });
        if (state == 0 && adapter) continue;
        const Tensor& source = value.template Get<Tensor>();
        OrtValue input;
        Tensor::InitOrtValue(source.DataType(), source.Shape(), allocator, input);
        ASSERT_STATUS_OK(session.GetDataTransferManager().CopyTensor(source, *input.GetMutable<Tensor>()));
        ASSERT_STATUS_OK(bindings[state]->BindInput(name, input));
      }
      const auto& reference = expected[state == 0 ? 0 : 5].Get<Tensor>();
      Tensor::InitOrtValue(reference.DataType(), reference.Shape(), allocator, outputs[state]);
      ASSERT_STATUS_OK(bindings[state]->BindOutput(builder.output_names_[0], outputs[state]));
      ASSERT_STATUS_OK(bindings[state]->SynchronizeInputs());
    }
    OrtValue actual;
    const auto& reference = expected[0].Get<Tensor>();
    Tensor::InitOrtValue(reference.DataType(), reference.Shape(), CPUAllocator::DefaultInstance(), actual);
    // Each adapter state owns a capture ID and stable GPU bindings. Replay
    // skips branch selection, so one ID cannot be reused across on/off states.
    for (int state : {0, 0, 0, 1, 1, 1, 0, 1}) {
      SCOPED_TRACE(MakeString("capture state=", state));
      auto* cpu_output = actual.GetMutable<Tensor>();
      auto values = cpu_output->MutableDataAsSpan<T>();
      std::fill(values.begin(), values.end(), T(-99.0f));
      ASSERT_STATUS_OK(session.GetDataTransferManager().CopyTensor(*cpu_output, *outputs[state].GetMutable<Tensor>()));
      RunOptions run_options;
      ASSERT_STATUS_OK(run_options.config_options.AddConfigEntry(kOrtRunOptionsConfigCudaGraphAnnotation,
                                                                 state == 0 ? "1" : "2"));
      ASSERT_STATUS_OK(session.Run(run_options, *bindings[state]));
      ASSERT_STATUS_OK(bindings[state]->SynchronizeOutputs());
      ASSERT_STATUS_OK(session.GetDataTransferManager().CopyTensor(outputs[state].Get<Tensor>(), *cpu_output));
      const auto comparison = CompareOrtValue(actual, expected[state == 0 ? 0 : 5], 0.0, 0.0, false);
      EXPECT_EQ(comparison.first, COMPARE_RESULT::SUCCESS) << comparison.second;
    }
    EXPECT_TRUE(capture_provider->IsGraphCaptured(1));
    EXPECT_TRUE(capture_provider->IsGraphCaptured(2));
  }
}

TEST_F(GraphTransformationTests, MatMulNBitsQLoRAFusionSerializedEmptyDefaults) {
  RunQLoRASerializedEmptyDefaults<MLFloat16>(*logger_);
}

TEST_F(GraphTransformationTests, MatMulNBitsQLoRAFusionSerializedEmptyDefaultsFp32) {
  RunQLoRASerializedEmptyDefaults<float>(*logger_);
}

TEST_F(GraphTransformationTests, MatMulNBitsQLoRAFusionJointDP4ACapture) {
  RunQLoRASerializedEmptyDefaults<float>(*logger_, 2048);
}

#endif
}  // namespace onnxruntime::test
