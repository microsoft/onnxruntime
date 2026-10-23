// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <map>
#include <sstream>

#include "gtest/gtest.h"
#include "core/framework/data_transfer.h"
#include "core/framework/tensor.h"
#include "core/graph/model.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime::test {

TEST(SpeculativePrefixTest, CudaGraphCaptureAndReplay) {
  auto ep = CudaExecutionProviderWithOptions(
      ProviderOptions{{"enable_cuda_graph", "1"}, {"min_num_runs_before_cuda_graph_capture", "4"}});
  if (!ep) GTEST_SKIP() << "CUDA is required";
  auto cpu_allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  Model model("speculative_prefix_capture", true, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 21}, {kMSDomain, 1}},
              {}, DefaultLoggingManager().DefaultLogger(), ModelOptions(true, true));
  Graph& graph = model.MainGraph();
  NodeArg& empty = graph.GetOrCreateNodeArg("", nullptr);
  std::map<std::string, OrtValue> cpu_inputs;
  auto input = [&]<typename T>(const char* name, std::vector<int64_t> shape, const std::vector<T>& values) {
    ONNX_NAMESPACE::TypeProto type;
    type.mutable_tensor_type()->set_elem_type(std::is_same_v<T, float>
                                                  ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT
                                                  : ONNX_NAMESPACE::TensorProto_DataType_INT32);
    for (int64_t dim : shape) type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
    Tensor tensor(DataTypeImpl::GetType<T>(), TensorShape(shape), cpu_allocator);
    std::copy(values.begin(), values.end(), tensor.MutableData<T>());
    Tensor::InitOrtValue(std::move(tensor), cpu_inputs[name]);
    return &graph.GetOrCreateNodeArg(name, &type);
  };
  std::vector<const NodeArg*> graph_outputs;
  auto output = [&](const std::string& name, bool integer = false) {
    ONNX_NAMESPACE::TypeProto type;
    type.mutable_tensor_type()->set_elem_type(integer ? ONNX_NAMESPACE::TensorProto_DataType_INT32
                                                      : ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto* arg = &graph.GetOrCreateNodeArg(name, &type);
    graph_outputs.push_back(arg);
    return arg;
  };
  const std::vector<float> rows{0, 1, 2, 3, 4, 5, 6, 7};
  auto& conv = graph.AddNode(
      "conv", "CausalConvWithState", "",
      {input("conv_x", {1, 4, 2}, rows), input("conv_w", {2, 1, 2}, std::vector<float>(4, 1)),
       &empty, input("conv_past", {1, 2, 3}, std::vector<float>(6, 0))},
      {output("conv_y"), output("conv_final"), output("conv_mid1"), output("conv_mid2"), output("conv_mid3")},
      nullptr, kMSDomain);
  conv.AddAttribute("channels_last", int64_t{1});
  conv.AddAttribute("state_channels_last", int64_t{0});
  conv.AddAttribute("state_length", int64_t{3});
  auto* cu = input("cu", {2}, std::vector<int32_t>{0, 4});
  auto& gdn = graph.AddNode(
      "gdn", "GatedDeltaNet", "",
      {input("gdn_q", {4, 1, 2}, std::vector<float>(8, 0)), input("gdn_k", {4, 1, 2}, rows),
       input("gdn_v", {4, 1, 2}, std::vector<float>(8, 1)), cu, &empty, &empty,
       input("gdn_past", {1, 1, 2, 2}, std::vector<float>(4, 0))},
      {output("gdn_y"), output("gdn_final"), output("gdn_mid1"), output("gdn_mid2"), output("gdn_mid3")},
      nullptr, kMSDomain);
  gdn.AddAttribute("state_output", std::string{"mid_state"});
  gdn.AddAttribute("update_rule", std::string{"linear"});
  auto* norm = input("norm", {2}, std::vector<float>(2, 1));
  auto& qsa = graph.AddNode(
      "qsa", "PackedSparseAttentionIndexer", "",
      {input("qsa_q", {4, 2}, std::vector<float>(8, 0)), input("qsa_k", {4, 2}, rows), norm, norm,
       input("cos", {32, 2}, std::vector<float>(64, 1)), input("sin", {32, 2}, std::vector<float>(64, 0)),
       cu, input("past_length", {1}, std::vector<int32_t>{0}), &empty, &empty, &empty, &empty,
       input("qsa_keys", {1, 4, 2}, std::vector<float>(8, 0)),
       input("qsa_buffer", {1, 3, 2}, std::vector<float>(6, 0)), &empty,
       input("qsa_lengths", {1, 2}, std::vector<int32_t>{0, 0})},
      {output("selected", true), output("counts", true), output("qsa_final_keys"), output("qsa_final_buffer"),
       &empty, output("qsa_final_lengths", true), &empty, &empty,
       output("qsa_mid1"), output("qsa_len1", true), output("qsa_status", true),
       output("qsa_mid2"), output("qsa_len2", true), output("qsa_mid3"), output("qsa_len3", true)},
      nullptr, kMSDomain);
  qsa.AddAttribute("policy_mode", std::string{"qsa"});
  qsa.AddAttribute("compress_ratio", int64_t{2});
  qsa.AddAttribute("state_capacity", int64_t{4});
  qsa.AddAttribute("token_budget", int64_t{2});
  graph.SetOutputs(graph_outputs);
  ASSERT_STATUS_OK(graph.Resolve());
  std::string serialized;
  ASSERT_TRUE(model.ToProto().SerializeToString(&serialized));
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  InferenceSession session(options, GetEnvironment());
  IExecutionProvider* ep_ptr = ep.get();
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(ep)));
  std::istringstream stream(serialized);
  ASSERT_STATUS_OK(session.Load(stream));
  ASSERT_STATUS_OK(session.Initialize());
  auto allocators = ep_ptr->CreatePreferredAllocators();
  const auto gpu = std::find_if(allocators.begin(), allocators.end(), [](const auto& allocator) {
    return allocator->Info().device.Type() == OrtDevice::GPU &&
           allocator->Info().alloc_type != OrtAllocatorType::OrtReadOnlyAllocator &&
           allocator->Info().mem_type == OrtMemTypeDefault;
  });
  ASSERT_NE(gpu, allocators.end());
  auto allocator = session.GetAllocator((*gpu)->Info());
  ASSERT_NE(allocator, nullptr);
  std::map<std::string, OrtValue> device_inputs, device_outputs;
  std::unique_ptr<IOBinding> binding;
  ASSERT_STATUS_OK(session.NewIOBinding(&binding));
  for (const auto& [name, value] : cpu_inputs) {
    Tensor tensor(value.Get<Tensor>().DataType(), value.Get<Tensor>().Shape(), allocator);
    ASSERT_STATUS_OK(ep_ptr->GetDataTransfer()->CopyTensor(value.Get<Tensor>(), tensor));
    Tensor::InitOrtValue(std::move(tensor), device_inputs[name]);
    ASSERT_STATUS_OK(binding->BindInput(name, device_inputs.at(name)));
  }
  for (const auto* arg : graph_outputs) {
    const auto* shape = arg->Shape();
    ASSERT_NE(shape, nullptr);
    std::vector<int64_t> dims;
    for (const auto& dim : shape->dim()) {
      ASSERT_TRUE(dim.has_dim_value()) << arg->Name();
      dims.push_back(dim.dim_value());
    }
    const auto dtype = arg->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32
                           ? DataTypeImpl::GetType<int32_t>() : DataTypeImpl::GetType<float>();
    Tensor tensor(dtype, TensorShape(dims), allocator);
    Tensor::InitOrtValue(std::move(tensor), device_outputs[arg->Name()]);
    ASSERT_STATUS_OK(binding->BindOutput(arg->Name(), device_outputs.at(arg->Name())));
  }
  RunOptions run;
  ASSERT_STATUS_OK(run.config_options.AddConfigEntry("gpu_graph_id", "1"));
  for (int iteration = 0; iteration < 7; ++iteration) {
    if (iteration == 6) {
      ASSERT_TRUE(ep_ptr->IsGraphCaptured(1));
      for (const char* name : {"conv_x", "gdn_k", "qsa_k"}) {
        auto& cpu = *cpu_inputs.at(name).GetMutable<Tensor>();
        for (float& value : cpu.MutableDataAsSpan<float>()) value += 10;
        ASSERT_STATUS_OK(ep_ptr->GetDataTransfer()->CopyTensor(cpu, *device_inputs.at(name).GetMutable<Tensor>()));
      }
    }
    ASSERT_STATUS_OK(session.Run(run, *binding));
  }
  ASSERT_STATUS_OK(binding->SynchronizeOutputs());
  auto floats = [&](const std::string& name) {
    const Tensor& actual = device_outputs.at(name).Get<Tensor>();
    Tensor host(DataTypeImpl::GetType<float>(), actual.Shape(), cpu_allocator);
    ORT_THROW_IF_ERROR(ep_ptr->GetDataTransfer()->CopyTensor(actual, host));
    return std::vector<float>(host.DataAsSpan<float>().begin(), host.DataAsSpan<float>().end());
  };
  for (int prefix = 1; prefix <= 3; ++prefix) {
    std::vector<float> conv_expected(6, 0), gdn_expected(4, 0), qsa_expected(6, 0);
    for (int channel = 0; channel < 2; ++channel) {
      for (int row = 0; row < prefix; ++row)
        conv_expected[channel * 3 + 3 - prefix + row] = 10.0f + row * 2 + channel;
      float sum = 0;
      for (int row = 0; row < prefix; ++row) sum += 10.0f + row * 2 + channel;
      for (int value = 0; value < 2; ++value) gdn_expected[value * 2 + channel] = sum;
      if (prefix % 2 == 1) qsa_expected[channel] = 10.0f + (prefix - 1) * 2 + channel;
    }
    EXPECT_EQ(floats("conv_mid" + std::to_string(prefix)), conv_expected);
    EXPECT_EQ(floats("gdn_mid" + std::to_string(prefix)), gdn_expected);
    EXPECT_EQ(floats("qsa_mid" + std::to_string(prefix)), qsa_expected);
  }
}

}  // namespace onnxruntime::test
