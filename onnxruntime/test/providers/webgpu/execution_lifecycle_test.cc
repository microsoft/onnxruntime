// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include <limits>
#include <cmath>
#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#if defined(USE_WEBGPU)
#include "core/session/inference_session.h"
#include "test/test_environment.h"
#endif
namespace onnxruntime::test {
#ifdef USE_WEBGPU
namespace {
std::unique_ptr<IExecutionProvider> CreateWebGpuProvider(bool nchw = false) {
  ConfigOptions options;
  ORT_THROW_IF_ERROR(options.AddConfigEntry("ep.webgpuexecutionprovider.validationMode", "full"));
  ORT_THROW_IF_ERROR(options.AddConfigEntry("ep.webgpuexecutionprovider.storageBufferCacheMode", "disabled"));
  ORT_THROW_IF_ERROR(options.AddConfigEntry("ep.webgpuexecutionprovider.enableInt64", "1"));
  if (nchw) {
    ORT_THROW_IF_ERROR(options.AddConfigEntry("ep.webgpuexecutionprovider.preferredLayout", "NCHW"));
  }
  return WebGpuExecutionProviderWithOptions(options);
}
void RunWebGpu(OpTester& test, std::unique_ptr<IExecutionProvider> provider) {
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  options.graph_optimization_level = TransformerLevel::Default;
  test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
}
}  // namespace

TEST(WebGpuLifecycleTest, ConcatEmptyChunk) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Concat", 13);
  test.AddAttribute<int64_t>("axis", 0);
  for (int i = 0; i < 128; ++i) test.AddInput<float>(("empty" + std::to_string(i)).c_str(), {0}, {});
  test.AddInput<float>("X", {1}, {7});
  test.AddOutput<float>("Y", {1}, {7});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuLifecycleTest, SoftmaxEmptyAxis) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Softmax", 13);
  test.AddInput<float>("X", {1, 0}, {});
  test.AddOutput<float>("Y", {1, 0}, {});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuLifecycleTest, ConcatEmptyChunksBetweenAndAfterValues) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("Concat", 13);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddInput<float>("first", {1}, {3});
  for (int i = 0; i < 128; ++i) test.AddInput<float>(("middle" + std::to_string(i)).c_str(), {0}, {});
  test.AddInput<float>("last", {1}, {7});
  for (int i = 0; i < 128; ++i) test.AddInput<float>(("tail" + std::to_string(i)).c_str(), {0}, {});
  test.AddOutput<float>("Y", {2}, {3, 7});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuLifecycleTest, EmptySoftmaxStillValidatesAxis) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("Softmax", 13);
  test.AddAttribute<int64_t>("axis", 3);
  test.AddInput<float>("X", {1, 0}, {});
  test.AddOutput<float>("Y", {1, 0}, {});
  test.Config(OpTester::ExpectResult::kExpectFailure, "axis");
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuLifecycleTest, NonEmptySoftmax) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("Softmax", 13);
  test.AddInput<float>("X", {1, 2}, {0, 0});
  test.AddOutput<float>("Y", {1, 2}, {0.5f, 0.5f});
  RunWebGpu(test, std::move(ep));
}

namespace {
void RunFailedCaptureRetry(const std::string& op) {
  SCOPED_TRACE(op);
  ConfigOptions config;
  ASSERT_STATUS_OK(config.AddConfigEntry("ep.webgpuexecutionprovider.validationMode", "full"));
  ASSERT_STATUS_OK(config.AddConfigEntry("ep.webgpuexecutionprovider.enableGraphCapture", "1"));
  auto ep = WebGpuExecutionProviderWithOptions(config);
  if (!ep) GTEST_SKIP();
  auto* provider = ep.get();
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);
  auto* graph = model.mutable_graph();
  graph->set_name("failed_capture");
  for (const char* name : {"A", "B"}) {
    auto* arg = graph->add_input();
    arg->set_name(name);
    auto* type = arg->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type->mutable_shape()->add_dim()->set_dim_param(std::string(name) + "0");
    type->mutable_shape()->add_dim()->set_dim_param(std::string(name) + "1");
    if (op == "Einsum" && std::string(name) == "A") {
      type->mutable_shape()->add_dim()->set_dim_param("A2");
    }
  }
  auto* out = graph->add_output();
  out->set_name("Y");
  auto* out_type = out->mutable_type()->mutable_tensor_type();
  out_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  out_type->mutable_shape()->add_dim()->set_dim_param("M");
  out_type->mutable_shape()->add_dim()->set_dim_param("N");
  auto* node = graph->add_node();
  node->set_name("failing_kernel");
  node->set_op_type(op);
  if (op == "Einsum") {
    auto* equation = node->add_attribute();
    equation->set_name("equation");
    equation->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
    equation->set_s("ijj,jk->ik");
  }
  node->add_input("A");
  node->add_input("B");
  node->add_output("Y");
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  InferenceSession session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(ep)));
  const auto bytes = model.SerializeAsString();
  ASSERT_STATUS_OK(session.Load(bytes.data(), static_cast<int>(bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());
  const auto allocators = provider->CreatePreferredAllocators();
  const auto allocator_it = std::find_if(allocators.begin(), allocators.end(), [](const auto& candidate) {
    return candidate->Info().device.Type() == OrtDevice::GPU && candidate->Info().alloc_type == OrtDeviceAllocator;
  });
  ASSERT_NE(allocator_it, allocators.end());
  const auto allocator = *allocator_it;
  OrtValue a, b;
  const TensorShape a_shape = op == "Einsum" ? TensorShape{2, 3, 3} : TensorShape{2, 3};
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), a_shape, allocator, a);
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{4, 2}, allocator, b);
  std::vector<OrtValue> outputs;
  const std::vector<std::string> output_names{"Y"};
  const auto status = session.Run(RunOptions{}, {{"A", a}, {"B", b}}, output_names, &outputs);
  ASSERT_FALSE(status.IsOK());
  EXPECT_FALSE(provider->IsGraphCaptured(0)) << status.ErrorMessage();
  const auto retry_status = session.Run(RunOptions{}, {{"A", a}, {"B", b}}, output_names, &outputs);
  EXPECT_FALSE(retry_status.IsOK()) << "An identical invalid run must not become successful replay.";
  EXPECT_FALSE(provider->IsGraphCaptured(0));

  Tensor cpu_a(DataTypeImpl::GetType<float>(), a_shape, CPUAllocator::DefaultInstance());
  Tensor cpu_b(DataTypeImpl::GetType<float>(), TensorShape{3, 2}, CPUAllocator::DefaultInstance());
  std::fill_n(cpu_a.MutableData<float>(), a_shape.Size(), 1.f);
  std::fill_n(cpu_b.MutableData<float>(), 6, 1.f);
  OrtValue valid_b;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{3, 2}, allocator, valid_b);
  ASSERT_STATUS_OK(session.GetDataTransferManager().CopyTensor(cpu_a, *a.GetMutable<Tensor>()));
  ASSERT_STATUS_OK(session.GetDataTransferManager().CopyTensor(cpu_b, *valid_b.GetMutable<Tensor>()));
  ASSERT_STATUS_OK(session.Run(RunOptions{}, {{"A", a}, {"B", valid_b}}, output_names, &outputs));
  EXPECT_TRUE(provider->IsGraphCaptured(0));
  ASSERT_EQ(outputs.size(), 1u);
  const auto& result = outputs[0].Get<Tensor>();
  ASSERT_EQ(result.Shape(), TensorShape({2, 2}));
  for (size_t i = 0; i < 4; ++i) EXPECT_EQ(result.Data<float>()[i], 3.f);

#if !defined(ORT_USE_EP_API_ADAPTERS)
  // Plugin RunOptions currently has no termination query in the public EP API.
  ASSERT_STATUS_OK(provider->ReleaseCapturedGraph(0));
  RunOptions cancelled;
  cancelled.terminate = true;
  EXPECT_FALSE(session.Run(cancelled, {{"A", a}, {"B", valid_b}}, output_names, &outputs).IsOK());
  EXPECT_FALSE(provider->IsGraphCaptured(0));
  std::fill_n(cpu_a.MutableData<float>(), a_shape.Size(), 2.f);
  ASSERT_STATUS_OK(session.GetDataTransferManager().CopyTensor(cpu_a, *a.GetMutable<Tensor>()));
  ASSERT_STATUS_OK(session.Run(RunOptions{}, {{"A", a}, {"B", valid_b}}, output_names, &outputs));
  EXPECT_TRUE(provider->IsGraphCaptured(0));
  ASSERT_EQ(outputs.size(), 1u);
  for (size_t i = 0; i < 4; ++i) EXPECT_EQ(outputs[0].Get<Tensor>().Data<float>()[i], 6.f);
#endif
}
}  // namespace

TEST(WebGpuLifecycleTest, FailedStatusMustNotBecomeCapturedGraph) {
  RunFailedCaptureRetry("Gemm");
}

TEST(WebGpuLifecycleTest, FailedExceptionMustNotBecomeCapturedGraph) {
  RunFailedCaptureRetry("Einsum");
}

#endif
}  // namespace onnxruntime::test
