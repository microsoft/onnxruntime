// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "core/framework/tensor.h"
#include "core/graph/model.h"
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

float FirstValue(const OrtValue& value) {
  return static_cast<float>(value.Get<Tensor>().Data<MLFloat16>()[0]);
}

void RunAdapterSwitchTest(bool qkv) {
  if (!DefaultWebGpuExecutionProvider()) GTEST_SKIP() << "WebGPU EP unavailable in this build.";

  const int projection_count = qkv ? 3 : 2;
  Model model("qlora_fused_adapter_switch", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 21}, {kMSDomain, 1}}, {},
              DefaultLoggingManager().DefaultLogger());
  ModelTestBuilder builder(model.MainGraph());
  NodeArg* x = builder.MakeInput<MLFloat16>({1, 32}, MLFloat16(0.5f), MLFloat16(0.5f));
  NodeArg* norm_scale = builder.MakeInitializer<MLFloat16>({32}, MLFloat16(1.0f), MLFloat16(1.0f));
  NodeArg* normalized = builder.MakeIntermediate<MLFloat16>(std::vector<int64_t>{1, 32});
  Node& norm = builder.AddNode("SimplifiedLayerNormalization", {x, norm_scale}, {normalized});
  norm.AddAttribute("epsilon", 1e-6f);
  norm.SetExecutionProviderType(kWebGpuExecutionProvider);

  std::vector<AdapterInputs> adapters;
  std::vector<NodeArg*> projections;
  for (int projection = 0; projection < projection_count; ++projection) {
    const int64_t n = qkv && projection > 0 ? 4 : 8;
    NodeArg* base_weights = builder.MakeInitializer<uint8_t>({n, 1, 16}, std::vector<uint8_t>(n * 16, 0x99));
    NodeArg* base_scales = builder.MakeInitializer<MLFloat16>({n, 1}, MLFloat16(0.03125f), MLFloat16(0.03125f));
    NodeArg* down_weights = builder.MakeInput<uint8_t>({32, 1, 32}, std::vector<uint8_t>(1024, 129));
    NodeArg* down_scales = builder.MakeInput<MLFloat16>({32, 1}, MLFloat16(0.03125f), MLFloat16(0.03125f));
    NodeArg* up_weights = builder.MakeInput<uint8_t>({n, 1, 32}, std::vector<uint8_t>(n * 32, 129));
    NodeArg* up_scales = builder.MakeInput<MLFloat16>({n, 1}, MLFloat16(0.0625f), MLFloat16(0.0625f));
    auto dynamic_shape = *up_weights->Shape();
    dynamic_shape.mutable_dim(1)->Clear();
    up_weights->SetShape(dynamic_shape);
    adapters.push_back({down_weights->Name(), down_scales->Name(), up_weights->Name(), up_scales->Name(), n});

    NodeArg* output = qkv ? builder.MakeOutput<MLFloat16>(std::vector<int64_t>{1, n})
                          : builder.MakeIntermediate<MLFloat16>(std::vector<int64_t>{1, n});
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
    projections.push_back(output);
  }
  if (!qkv) {
    NodeArg* sigmoid_output = builder.MakeIntermediate<MLFloat16>(std::vector<int64_t>{1, 8});
    NodeArg* activated = builder.MakeIntermediate<MLFloat16>(std::vector<int64_t>{1, 8});
    NodeArg* y = builder.MakeOutput<MLFloat16>(std::vector<int64_t>{1, 8});
    Node& sigmoid = builder.AddNode("Sigmoid", {projections[0]}, {sigmoid_output});
    Node& silu = builder.AddNode("Mul", {projections[0], sigmoid_output}, {activated});
    Node& output_mul = builder.AddNode("Mul", {activated, projections[1]}, {y});
    for (Node* node : {&sigmoid, &silu, &output_mul}) {
      node->SetExecutionProviderType(kWebGpuExecutionProvider);
    }
  }
  builder.SetGraphOutputs();
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

  std::vector<std::vector<OrtValue>> baseline_outputs;
  for (bool fused : {false, true}) {
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Level1;
    InferenceSessionWrapper session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultWebGpuExecutionProvider()));
    ASSERT_STATUS_OK(session.Load(model_data.data(), static_cast<int>(model_data.size())));
    if (fused) {
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
      fused_count += node.OpType() == (qkv ? "MatMulNBitsQkv" : "MatMulNBitsMlp");
    }
    ASSERT_EQ(fused_count, fused ? 1u : 0u);

    std::vector<std::vector<OrtValue>> current_outputs;
    for (size_t state_index = 0; state_index < states.size(); ++state_index) {
      const auto [mask, changed] = states[state_index];
      SCOPED_TRACE(MakeString(qkv ? "QKV" : "MLP", ", fused=", fused,
                              ", mask=", mask, ", changed=", changed));
      NameMLValMap feeds = builder.feeds_;
      for (int projection = 0; projection < projection_count; ++projection) {
        const auto& adapter = adapters[projection];
        if ((mask & (1 << projection)) == 0) {
          feeds[adapter.up_weights] = FilledCpuTensor<uint8_t>({adapter.n, 0, 32}, 0);
        } else if (changed) {
          feeds[adapter.down_weights] = FilledCpuTensor<uint8_t>({32, 1, 32}, 130);
          feeds[adapter.down_scales] = FilledCpuTensor<MLFloat16>({32, 1}, MLFloat16(0.0625f));
          feeds[adapter.up_weights] = FilledCpuTensor<uint8_t>({adapter.n, 1, 32}, 130);
          feeds[adapter.up_scales] = FilledCpuTensor<MLFloat16>({adapter.n, 1}, MLFloat16(0.125f));
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
      const size_t output = qkv ? static_cast<size_t>(projection) : 0u;
      EXPECT_GT(std::abs(FirstValue(current_outputs[projection + 1][output]) -
                         FirstValue(current_outputs[0][output])),
                0.1f);
    }
    const size_t changed_index = static_cast<size_t>(projection_count) + 2;
    EXPECT_GT(std::abs(FirstValue(current_outputs[changed_index][0]) -
                       FirstValue(current_outputs[changed_index - 1][0])),
              0.1f);
    for (size_t output = 0; output < current_outputs.front().size(); ++output) {
      const auto comparison = CompareOrtValue(current_outputs.back()[output], current_outputs.front()[output],
                                              1e-3, 1e-3, false);
      EXPECT_EQ(comparison.first, COMPARE_RESULT::SUCCESS) << comparison.second;
    }
    if (!fused) baseline_outputs = std::move(current_outputs);
  }
}

}  // namespace

TEST(MatMulNBitsQLoRAFusedAdapterTest, MlpSwitchesInOneSession) { RunAdapterSwitchTest(false); }
TEST(MatMulNBitsQLoRAFusedAdapterTest, QkvSwitchesInOneSession) { RunAdapterSwitchTest(true); }

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace onnxruntime::test
