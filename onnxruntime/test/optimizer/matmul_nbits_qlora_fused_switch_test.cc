// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "core/framework/tensor.h"
#include "core/graph/model.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/optimizer/matmul_nbits_mlp_fusion.h"
#include "core/optimizer/matmul_nbits_qkv_fusion.h"
#include "test/optimizer/webgpu_fusion_test_util.h"
#include "test/util/include/default_providers.h"
#include "test/unittest_util/framework_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime::test {

#if !defined(DISABLE_CONTRIB_OPS)
namespace {

struct AdapterInputs {
  std::string down_weights;
  std::string down_scales;
  std::string up_weights;
  std::string up_scales;
  int64_t n;
};

template <typename T>
OrtValue FilledCpuTensor(const TensorShape& shape, T value) {
  OrtValue result;
  Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), shape, CPUAllocator::DefaultInstance(), result);
  auto data = result.GetMutable<Tensor>()->MutableDataAsSpan<T>();
  std::fill(data.begin(), data.end(), value);
  return result;
}

void AddEmptyOverridableDefault(Graph& graph, NodeArg& arg, int empty_axis) {
  ONNX_NAMESPACE::TensorProto init;
  init.set_name(arg.Name());
  init.set_data_type(arg.TypeAsProto()->tensor_type().elem_type());
  const auto& original_shape = *arg.Shape();
  for (const auto& dim : original_shape.dim()) init.add_dims(dim.dim_value());
  init.set_dims(empty_axis, 0);
  graph.AddInitializedTensor(init);
  auto dynamic_shape = original_shape;
  dynamic_shape.mutable_dim(empty_axis)->Clear();
  arg.SetShape(dynamic_shape);
}

template <typename T>
float FirstValue(const OrtValue& value) {
  return static_cast<float>(value.Get<Tensor>().Data<T>()[0]);
}

struct SwitchTestConfig {
  bool raw = false;
  bool skip = false;
  int64_t batch = 1;
  int64_t rows = 1;
  bool defaults = false;
};

template <typename T>
void RunAdapterSwitchTest(bool qkv, SwitchTestConfig config = {}) {
  if (!DefaultWebGpuExecutionProvider()) GTEST_SKIP() << "WebGPU EP unavailable in this build.";

  const int projection_count = qkv ? 3 : 2;
  Model model("qlora_fused_adapter_switch", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 21}, {kMSDomain, 1}}, {},
              DefaultLoggingManager().DefaultLogger());
  ModelTestBuilder builder(model.MainGraph());
  const std::vector<int64_t> input_shape{config.batch, config.rows, 32};
  // ModelTestBuilder uses a fixed seed; vary rows and channels reproducibly.
  NodeArg* x = builder.MakeInput<T>(input_shape, T(0.25f), T(0.75f));
  NodeArg* norm_scale = builder.MakeInitializer<T>({32}, T(1.0f), T(1.0f));
  NodeArg* normalized = builder.MakeIntermediate<T>(input_shape);
  NodeArg* skip = config.skip ? builder.MakeInput<T>(input_shape, T(-0.125f), T(0.125f)) : nullptr;
  std::vector<NodeArg*> graph_inputs{x};
  if (skip) graph_inputs.push_back(skip);
  NodeArg* residual = config.skip ? builder.MakeOutput<T>(input_shape) : nullptr;
  NodeArg* optional = builder.MakeOptionalTensor();
  std::vector<NodeArg*> norm_outputs{normalized};
  if (config.skip) norm_outputs.insert(norm_outputs.end(), {optional, optional, residual});
  Node& norm = config.skip
                   ? builder.AddNode("SkipSimplifiedLayerNormalization", {x, skip, norm_scale}, norm_outputs, kMSDomain)
                   : builder.AddNode("SimplifiedLayerNormalization", {x, norm_scale}, norm_outputs);
  norm.AddAttribute("epsilon", 1e-6f);
  norm.SetExecutionProviderType(kWebGpuExecutionProvider);

  std::vector<AdapterInputs> adapters;
  std::vector<NodeArg*> projections;
  for (int projection = 0; projection < projection_count; ++projection) {
    const int64_t n = qkv && projection > 0 ? 4 : 8;
    NodeArg* base_weights = builder.MakeInitializer<uint8_t>({n, 1, 16}, std::vector<uint8_t>(n * 16, 0x99));
    NodeArg* base_scales = builder.MakeInitializer<T>({n, 1}, T(0.03125f), T(0.03125f));
    NodeArg* down_weights = builder.MakeInput<uint8_t>({32, 1, 32}, std::vector<uint8_t>(1024, 129));
    NodeArg* down_scales = builder.MakeInput<T>({32, 1}, T(0.03125f), T(0.03125f));
    NodeArg* up_weights = builder.MakeInput<uint8_t>({n, 1, 32}, std::vector<uint8_t>(n * 32, 129));
    NodeArg* up_scales = builder.MakeInput<T>({n, 1}, T(0.0625f), T(0.0625f));
    graph_inputs.insert(graph_inputs.end(), {down_weights, down_scales, up_weights, up_scales});
    if (config.defaults) {
      AddEmptyOverridableDefault(builder.graph_, *down_weights, 0);
      AddEmptyOverridableDefault(builder.graph_, *down_scales, 0);
      AddEmptyOverridableDefault(builder.graph_, *up_weights, 1);
      AddEmptyOverridableDefault(builder.graph_, *up_scales, 1);
    }
    auto dynamic_shape = *up_weights->Shape();
    if (!config.raw) dynamic_shape.mutable_dim(0)->Clear();
    dynamic_shape.mutable_dim(1)->Clear();
    up_weights->SetShape(dynamic_shape);
    adapters.push_back({down_weights->Name(), down_scales->Name(), up_weights->Name(), up_scales->Name(), n});

    const std::vector<int64_t> output_shape{config.batch, config.rows, n};
    NodeArg* output = qkv ? builder.MakeOutput<T>(output_shape) : builder.MakeIntermediate<T>(output_shape);
    if (config.raw) {
      NodeArg* base = builder.MakeIntermediate<T>(output_shape);
      NodeArg* down = builder.MakeIntermediate<T>(std::vector<int64_t>{config.batch, config.rows, 32});
      NodeArg* delta = builder.MakeIntermediate<T>(output_shape);
      Node& base_node = builder.AddNode("MatMulNBits", {normalized, base_weights, base_scales}, {base}, kMSDomain);
      Node& down_node = builder.AddNode("MatMulNBits", {normalized, down_weights, down_scales}, {down}, kMSDomain);
      Node& up_node = builder.AddNode("MatMulNBits", {down, up_weights, up_scales}, {delta}, kMSDomain);
      for (Node* node : {&base_node, &down_node, &up_node}) {
        node->AddAttribute("K", int64_t{32});
        node->AddAttribute("N", node == &down_node ? int64_t{32} : n);
        node->AddAttribute("bits", node == &base_node ? int64_t{4} : int64_t{8});
        node->AddAttribute("block_size", int64_t{32});
        node->AddAttribute("accuracy_level", node == &base_node ? int64_t{4} : int64_t{0});
        node->SetExecutionProviderType(kWebGpuExecutionProvider);
      }
      builder.AddNode("Add", {base, delta}, {output}).SetExecutionProviderType(kWebGpuExecutionProvider);
    } else {
      Node& node = builder.AddNode("MatMulNBitsQLoRA",
                                   {normalized, base_weights, base_scales,
                                    down_weights, down_scales, up_weights, up_scales},
                                   {output}, kMSDomain);
      node.AddAttribute("K", int64_t{32});
      node.AddAttribute("N", n);
      node.AddAttribute("bits", int64_t{4});
      node.AddAttribute("block_size", int64_t{32});
      node.AddAttribute("lora_bits", int64_t{8});
      node.AddAttribute("lora_block_size", int64_t{32});
      node.SetExecutionProviderType(kWebGpuExecutionProvider);
    }
    projections.push_back(output);
  }
  if (!qkv) {
    const std::vector<int64_t> output_shape{config.batch, config.rows, 8};
    NodeArg* sigmoid_output = builder.MakeIntermediate<T>(output_shape);
    NodeArg* activated = builder.MakeIntermediate<T>(output_shape);
    NodeArg* y = builder.MakeOutput<T>(output_shape);
    Node& sigmoid = builder.AddNode("Sigmoid", {projections[0]}, {sigmoid_output});
    Node& silu = builder.AddNode("Mul", {projections[0], sigmoid_output}, {activated});
    Node& output_mul = builder.AddNode("Mul", {activated, projections[1]}, {y});
    for (Node* node : {&sigmoid, &silu, &output_mul}) {
      node->SetExecutionProviderType(kWebGpuExecutionProvider);
    }
  }
  builder.SetGraphOutputs();
  if (config.defaults) builder.graph_.SetInputs(graph_inputs);
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  std::string model_data;
  ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));

  // The graph and both sessions stay fixed while adapter B shapes and contents change.
  std::vector<std::pair<int, bool>> states{{0, false}};
  for (int projection = 0; projection < projection_count; ++projection) {
    states.emplace_back(1 << projection, false);
  }
  const int all_on = (1 << projection_count) - 1;
  states.emplace_back(all_on, false);
  states.emplace_back(all_on, true);
  states.emplace_back(0, false);
  if (config.defaults) states.emplace_back(-1, false);  // use the empty overridable defaults

  std::vector<std::vector<OrtValue>> baseline_outputs;
  for (bool fused : {false, true}) {
    SessionOptions options;
    options.graph_optimization_level = config.raw ? TransformerLevel::Level2 : TransformerLevel::Level1;
    if (config.raw && !fused) {
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableSpecifiedOptimizers,
                                                             "MatMulNBitsQLoRAFusion"));
    }
    InferenceSessionWrapper session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultWebGpuExecutionProvider()));
    ASSERT_STATUS_OK(session.Load(model_data.data(), static_cast<int>(model_data.size())));
    if (fused && !config.raw) {
      if (qkv) {
        ASSERT_STATUS_OK(session.RegisterGraphTransformer(
            std::make_unique<MatMulNBitsQkvFusion>(InlinedHashSet<std::string_view>{kWebGpuExecutionProvider}, true),
            TransformerLevel::Level2));
      } else {
        ASSERT_STATUS_OK(session.RegisterGraphTransformer(
            std::make_unique<MatMulNBitsMlpFusion>(InlinedHashSet<std::string_view>{kWebGpuExecutionProvider}, true),
            TransformerLevel::Level2));
      }
    }
    ASSERT_STATUS_OK(session.Initialize());
    size_t fused_count = 0;
    for (const auto& node : session.GetGraph().Nodes()) {
      if (node.OpType() == (qkv ? "MatMulNBitsQLoRAQkv" : "MatMulNBitsQLoRAMlp")) {
        ++fused_count;
        EXPECT_EQ(node.GetExecutionProviderType(), kWebGpuExecutionProvider);
      }
    }
    ASSERT_EQ(fused_count, fused ? 1u : 0u);
    if (config.defaults) {
      EXPECT_EQ(session.GetGraph().GetOverridableInitializers().size(), static_cast<size_t>(4 * projection_count));
    }

    std::vector<std::vector<OrtValue>> current_outputs;
    for (size_t state_index = 0; state_index < states.size(); ++state_index) {
      const auto [mask, changed] = states[state_index];
      SCOPED_TRACE(MakeString(qkv ? "QKV" : "MLP", ", fused=", fused,
                              ", mask=", mask, ", changed=", changed));
      NameMLValMap feeds = builder.feeds_;
      if (mask == -1) {
        for (const auto& adapter : adapters) {
          feeds.erase(adapter.down_weights);
          feeds.erase(adapter.down_scales);
          feeds.erase(adapter.up_weights);
          feeds.erase(adapter.up_scales);
        }
      }
      for (int projection = 0; mask != -1 && projection < projection_count; ++projection) {
        const auto& adapter = adapters[projection];
        if ((mask & (1 << projection)) == 0) {
          feeds[adapter.up_weights] = FilledCpuTensor<uint8_t>({adapter.n, 0, 32}, 0);
        } else if (changed) {
          feeds[adapter.down_weights] = FilledCpuTensor<uint8_t>({32, 1, 32}, 130);
          feeds[adapter.down_scales] = FilledCpuTensor<T>({32, 1}, T(0.0625f));
          feeds[adapter.up_weights] = FilledCpuTensor<uint8_t>({adapter.n, 1, 32}, 130);
          feeds[adapter.up_scales] = FilledCpuTensor<T>({adapter.n, 1}, T(0.125f));
        }
      }
      std::vector<OrtValue> outputs;
      ASSERT_STATUS_OK(session.Run(RunOptions{}, feeds, builder.output_names_, &outputs));
      if (fused) {
        ASSERT_EQ(outputs.size(), baseline_outputs[state_index].size());
        for (size_t output = 0; output < outputs.size(); ++output) {
          const auto comparison = CompareOrtValue(outputs[output], baseline_outputs[state_index][output],
                                                  2e-3, 1e-2, false);
          EXPECT_EQ(comparison.first, COMPARE_RESULT::SUCCESS) << comparison.second;
        }
      }
      current_outputs.push_back(std::move(outputs));
    }

    for (int projection = 0; projection < projection_count; ++projection) {
      const size_t output = qkv ? static_cast<size_t>(projection) + (config.skip ? 1u : 0u)
                                : (config.skip ? 1u : 0u);
      EXPECT_GT(std::abs(FirstValue<T>(current_outputs[projection + 1][output]) -
                         FirstValue<T>(current_outputs[0][output])),
                0.1f);
    }
    const size_t changed_index = static_cast<size_t>(projection_count) + 2;
    const size_t main_output = config.skip ? 1u : 0u;
    EXPECT_GT(std::abs(FirstValue<T>(current_outputs[changed_index][main_output]) -
                       FirstValue<T>(current_outputs[changed_index - 1][main_output])),
              0.1f);
    for (size_t output = 0; output < current_outputs.front().size(); ++output) {
      const auto comparison = CompareOrtValue(current_outputs.back()[output], current_outputs.front()[output],
                                              1e-3, 1e-3, false);
      EXPECT_EQ(comparison.first, COMPARE_RESULT::SUCCESS) << comparison.second;
    }
    // A malformed zero-element B must be rejected even when every projection
    // takes the empty-adapter shortcut. Also exercise an unrelated active B.
    for (int bad_projection = 0; !config.raw && bad_projection < projection_count; ++bad_projection) {
      for (bool mixed_active : {false, true}) {
        SCOPED_TRACE(MakeString(qkv ? "QKV" : "MLP", ", fused=", fused,
                                ", bad_projection=", bad_projection, ", mixed_active=", mixed_active));
        NameMLValMap feeds = builder.feeds_;
        for (const auto& adapter : adapters) {
          feeds[adapter.up_weights] = FilledCpuTensor<uint8_t>({adapter.n, 0, 32}, 0);
        }
        if (mixed_active) {
          const auto& active = adapters[(bad_projection + 1) % projection_count];
          feeds[active.up_weights] = builder.feeds_.at(active.up_weights);
        }
        feeds[adapters[bad_projection].up_weights] = FilledCpuTensor<uint8_t>({0, 1, 32}, 0);
        std::vector<OrtValue> outputs;
        const auto status = session.Run(RunOptions{}, feeds, builder.output_names_, &outputs);
        EXPECT_FALSE(status.IsOK());
        EXPECT_NE(status.ErrorMessage().find("Invalid packed adapter dimensions."), std::string::npos)
            << status.ErrorMessage();
      }
    }
    if (!fused) baseline_outputs = std::move(current_outputs);
  }
}

}  // namespace

TEST(MatMulNBitsQLoRAFusedAdapterTest, MlpSwitchesInOneSession) { RunAdapterSwitchTest<MLFloat16>(false); }
TEST(MatMulNBitsQLoRAFusedAdapterTest, QkvSwitchesInOneSession) { RunAdapterSwitchTest<MLFloat16>(true); }
TEST(MatMulNBitsQLoRAFusedAdapterTest, MlpRawPipelineFp32Batched) {
  RunAdapterSwitchTest<float>(false, {.raw = true, .skip = true, .batch = 2, .rows = 3, .defaults = true});
}
TEST(MatMulNBitsQLoRAFusedAdapterTest, QkvRawPipelineFp32Batched) {
  RunAdapterSwitchTest<float>(true, {.raw = true, .skip = true, .batch = 2, .rows = 3, .defaults = true});
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace onnxruntime::test
