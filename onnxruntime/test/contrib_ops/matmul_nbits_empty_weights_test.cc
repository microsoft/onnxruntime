// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include <limits>

#include "core/session/IOBinding.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {

class MatMulNBitsEmptyWeights : public ::testing::TestWithParam<bool> {
  void SetUp() override {
    if (GetParam() && !DefaultWebGpuExecutionProvider()) GTEST_SKIP() << "WebGPU unavailable.";
  }
};

template <typename T>
void RunEmptyCases(bool webgpu) {
  struct Case {
    int64_t bits, block_size;
    std::vector<int64_t> a_shape;
    int empty_axis;
    bool constant, bias;
  };
  const Case cases[] = {
      {8, 32, {2, 3, 64}, 0, false, false},  // down projection, batched
      {8, 32, {2, 64}, 1, true, false},      // up projection, constant empty B
      {4, 16, {64}, 1, true, true},          // vector A and bias
      {2, 64, {2, 64}, 2, false, true},      // empty packed-blob axis
      {4, 32, {0, 64}, 0, false, false},     // empty output
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(MakeString("bits=", c.bits, ", block_size=", c.block_size, ", axis=", c.empty_axis));
    OpTester test("MatMulNBits", 1, kMSDomain);
    test.AddAttribute<int64_t>("K", 64);
    test.AddAttribute<int64_t>("N", 3);
    test.AddAttribute<int64_t>("bits", c.bits);
    test.AddAttribute<int64_t>("block_size", c.block_size);
    test.AddInput<T>("A", c.a_shape,
                     std::vector<T>(TensorShape(c.a_shape).Size(), T(std::numeric_limits<float>::quiet_NaN())));
    std::vector<int64_t> b_shape{3, 64 / c.block_size, c.block_size * c.bits / 8};
    b_shape[c.empty_axis] = 0;
    test.AddInput<uint8_t>("B", b_shape, {}, c.constant);
    test.AddInput<T>("scales", {0}, {}, true);
    // Unused quantization inputs need neither their normal shapes nor data.
    if (!c.constant) {
      test.AddInput<uint8_t>("zero_points", {0}, {});
      test.AddInput<int32_t>("g_idx", {0}, {});
    } else {
      test.AddOptionalInputEdge<uint8_t>();
      test.AddOptionalInputEdge<int32_t>();
    }
    if (c.bias) test.AddInput<T>("bias", {3}, {T(1.0f), T(-2.0f), T(3.0f)}, c.constant);
    auto y_shape = c.a_shape;
    y_shape.back() = 3;
    std::vector<T> expected(TensorShape(y_shape).Size(), T(0.0f));
    const T bias[] = {T(1.0f), T(-2.0f), T(3.0f)};
    if (c.bias) {
      for (size_t i = 0; i < expected.size(); ++i) expected[i] = bias[i % 3];
    }
    test.AddOutput<T>("Y", y_shape, expected);
    test.SetOutputTolerance(0.0f, 0.0f);
    SessionOptions options;
    options.use_per_session_threads = false;
    if (webgpu) ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    auto ep = webgpu ? DefaultWebGpuExecutionProvider() : DefaultCpuExecutionProvider();
    ASSERT_NE(ep, nullptr);
    size_t packed_count = 0;
    test.Config(options).ConfigEp(std::move(ep)).RunWithConfig(&packed_count);
    EXPECT_EQ(packed_count, 0u);
  }
}

TEST_P(MatMulNBitsEmptyWeights, ZeroAndBiasFp32) { RunEmptyCases<float>(GetParam()); }
TEST_P(MatMulNBitsEmptyWeights, ZeroAndBiasFp16) { RunEmptyCases<MLFloat16>(GetParam()); }

TEST_P(MatMulNBitsEmptyWeights, RejectMalformedPackedShape) {
  for (bool constant : {false, true}) {
    for (const auto& shape : {std::vector<int64_t>{0, 2, 33}, std::vector<int64_t>{0, 64}}) {
      OpTester test("MatMulNBits", 1, kMSDomain);
      test.AddAttribute<int64_t>("K", 64);
      test.AddAttribute<int64_t>("N", 3);
      test.AddAttribute<int64_t>("bits", 8);
      test.AddAttribute<int64_t>("block_size", 32);
      test.AddInput<float>("A", {1, 64}, std::vector<float>(64, 1.0f));
      test.AddInput<uint8_t>("B", shape, {}, constant);
      test.AddInput<float>("scales", {0}, {}, constant);
      test.AddOutput<float>("Y", {1, 3}, std::vector<float>(3));
      SessionOptions options;
      options.use_per_session_threads = false;
      if (GetParam())
        ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
      test.Config(options)
          .Config(OpTester::ExpectResult::kExpectFailure, "MatMulNBits: empty B")
          .ConfigEp(GetParam() ? DefaultWebGpuExecutionProvider() : DefaultCpuExecutionProvider())
          .RunWithConfig();
    }
  }
}

template <typename T>
void RunSwitching(bool webgpu, bool capture, bool with_bias = true) {
  Model model("empty MatMulNBits defaults", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 21}, {kMSDomain, 1}}, {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  ModelTestBuilder builder(graph);
  auto* a = builder.MakeInput<T>({2, 64}, T(0.5f), T(0.5f));
  auto* b = builder.MakeInput<uint8_t>({3, 2, 32}, uint8_t{129}, uint8_t{130});
  auto* s = builder.MakeInput<T>({3, 2}, T(0.25f), T(0.25f));
  for (auto* arg : {b, s}) {
    ONNX_NAMESPACE::TensorProto init;
    init.set_name(arg->Name());
    init.set_data_type(arg == b ? ONNX_NAMESPACE::TensorProto::UINT8 : utils::ToTensorProtoElementType<T>());
    init.add_dims(3);
    init.add_dims(0);
    if (arg == b) init.add_dims(32);
    graph.AddInitializedTensor(init);
    auto shape = *arg->Shape();
    shape.mutable_dim(1)->Clear();
    arg->SetShape(shape);
  }
  graph.SetInputs({a, b, s});
  auto* y = builder.MakeOutput<T>(std::vector<int64_t>{2, 3});
  std::vector<NodeArg*> inputs{a, b, s};
  if (with_bias) {
    auto* bias = builder.MakeInitializer<T>({3}, T(1.0f), T(1.0f));
    inputs.insert(inputs.end(), {builder.MakeEmptyInput(), builder.MakeEmptyInput(), bias});
  }
  auto& node = builder.AddNode("MatMulNBits", inputs, {y}, kMSDomain);
  node.AddAttribute("K", int64_t{64});
  node.AddAttribute("N", int64_t{3});
  node.AddAttribute("bits", int64_t{8});
  node.AddAttribute("block_size", int64_t{32});
  node.AddAttribute("accuracy_level", int64_t{0});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  ConfigOptions config;
  if (capture) ASSERT_STATUS_OK(config.AddConfigEntry("ep.webgpuexecutionprovider.enableGraphCapture", "1"));
  auto ep = webgpu ? WebGpuExecutionProviderWithOptions(config) : DefaultCpuExecutionProvider();
  ASSERT_NE(ep, nullptr);
  auto* provider = ep.get();
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  if (webgpu) ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  InferenceSession session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(ep)));
  const auto data = model.ToProto().SerializeAsString();
  ASSERT_STATUS_OK(session.Load(data.data(), static_cast<int>(data.size())));
  ASSERT_STATUS_OK(session.Initialize());

  auto off_feeds = builder.feeds_;
  off_feeds.erase(b->Name());
  off_feeds.erase(s->Name());
  auto check = [with_bias](const OrtValue& output, int state) {
    const T expected((with_bias ? 1.0f : 0.0f) + (state ? 8.0f : 0.0f));
    for (T value : output.Get<Tensor>().DataAsSpan<T>()) EXPECT_EQ(value, expected);
  };
  if (!capture) {
    for (int state : {0, 1, 0, 1}) {
      std::vector<OrtValue> output;
      ASSERT_STATUS_OK(session.Run(RunOptions{}, state ? builder.feeds_ : off_feeds, builder.output_names_, &output));
      check(output[0], state);
    }
    return;
  }

  const auto allocator = provider->CreatePreferredAllocators().back();
  ASSERT_NE(allocator->Info().device.Type(), OrtDevice::CPU);
  std::unique_ptr<IOBinding> bindings[2];
  OrtValue outputs[2];
  for (int state : {0, 1}) {
    ASSERT_STATUS_OK(session.NewIOBinding(&bindings[state]));
    for (const auto& [name, value] : state ? builder.feeds_ : off_feeds) {
      const auto& source = value.template Get<Tensor>();
      OrtValue input;
      Tensor::InitOrtValue(source.DataType(), source.Shape(), allocator, input);
      ASSERT_STATUS_OK(session.GetDataTransferManager().CopyTensor(source, *input.GetMutable<Tensor>()));
      ASSERT_STATUS_OK(bindings[state]->BindInput(name, input));
    }
    Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), TensorShape({2, 3}), allocator, outputs[state]);
    ASSERT_STATUS_OK(bindings[state]->BindOutput(y->Name(), outputs[state]));
    ASSERT_STATUS_OK(bindings[state]->SynchronizeInputs());
  }
  OrtValue actual;
  Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), TensorShape({2, 3}), CPUAllocator::DefaultInstance(), actual);
  for (int state : {0, 0, 0, 1, 1, 1, 0, 1}) {
    auto* cpu = actual.GetMutable<Tensor>();
    auto values = cpu->MutableDataAsSpan<T>();
    std::fill(values.begin(), values.end(), T(-99.0f));
    ASSERT_STATUS_OK(session.GetDataTransferManager().CopyTensor(*cpu, *outputs[state].GetMutable<Tensor>()));
    RunOptions run;
    ASSERT_STATUS_OK(run.config_options.AddConfigEntry(kOrtRunOptionsConfigCudaGraphAnnotation, state ? "2" : "1"));
    ASSERT_STATUS_OK(session.Run(run, *bindings[state]));
    ASSERT_STATUS_OK(bindings[state]->SynchronizeOutputs());
    ASSERT_STATUS_OK(session.GetDataTransferManager().CopyTensor(outputs[state].Get<Tensor>(), *cpu));
    check(actual, state);
  }
  EXPECT_TRUE(provider->IsGraphCaptured(1));
  EXPECT_TRUE(provider->IsGraphCaptured(2));
}

TEST_P(MatMulNBitsEmptyWeights, OverridesInSameSessionFp32) { RunSwitching<float>(GetParam(), false); }
TEST_P(MatMulNBitsEmptyWeights, OverridesInSameSessionFp16) { RunSwitching<MLFloat16>(GetParam(), false); }

INSTANTIATE_TEST_SUITE_P(Cpu, MatMulNBitsEmptyWeights, ::testing::Values(false));
#if defined(USE_WEBGPU)
INSTANTIATE_TEST_SUITE_P(WebGpu, MatMulNBitsEmptyWeights, ::testing::Values(true));
TEST(MatMulNBitsEmptyWeightsCapture, ZeroAndBiasReplay) {
  if (!DefaultWebGpuExecutionProvider()) GTEST_SKIP() << "WebGPU unavailable.";
  for (bool bias : {false, true}) {
    RunSwitching<float>(true, true, bias);
    RunSwitching<MLFloat16>(true, true, bias);
  }
}
#endif

}  // namespace
}  // namespace onnxruntime::test
#endif
