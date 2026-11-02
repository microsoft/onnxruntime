// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "test/test_environment.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

// Builds a FusedConv with the given activation attributes and returns the Initialize() status.
Status InitializeFusedConv(const std::string& activation, const std::vector<float>& activation_params,
                           std::unique_ptr<IExecutionProvider> webgpu_ep) {
  constexpr int64_t kChannels = 4;
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model("ConvUnknownActivation", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 17}, {kMSDomain, 1}}, {}, logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  auto* input = builder.MakeInput<float>({1, kChannels, 8}, -1.0f, 1.0f);
  auto* weight = builder.MakeInitializer<float>({kChannels, kChannels, 3}, -1.0f, 1.0f);
  auto* output = builder.MakeOutput();
  Node& conv = builder.AddNode("FusedConv", {input, weight}, {output}, kMSDomain);
  conv.AddAttribute("kernel_shape", std::vector<int64_t>{3});
  conv.AddAttribute("pads", std::vector<int64_t>{1, 1});
  conv.AddAttribute("strides", std::vector<int64_t>{1});
  conv.AddAttribute("dilations", std::vector<int64_t>{1});
  conv.AddAttribute("group", static_cast<int64_t>(1));
  conv.AddAttribute("activation", activation);
  if (!activation_params.empty()) {
    conv.AddAttribute("activation_params", activation_params);
  }

  builder.SetGraphOutputs();
  ORT_RETURN_IF_ERROR(graph.Resolve());

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  SessionOptions so;
  so.session_logid = "Conv_WebGPU.FusedActivation";
  InferenceSession session{so, GetEnvironment()};
  ORT_RETURN_IF_ERROR(session.RegisterExecutionProvider(std::move(webgpu_ep)));
  ORT_RETURN_IF_ERROR(session.Load(model_data.data(), static_cast<int>(model_data.size())));
  return session.Initialize();
}

}  // namespace

// ConvActivationFusion writes the attribute from the core library while the activation set lives
// in the EP, so a core library newer than the EP binary can fuse an activation the kernel has
// never heard of. That has to report the activation by name, not a bare failed assertion.
TEST(Conv_WebGPU, UnknownFusedActivationIsReportedByName) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  const Status status = InitializeFusedConv("NotAnActivation", {}, std::move(webgpu_ep));
  ASSERT_FALSE(status.IsOK()) << "the WebGPU EP has no shader for 'NotAnActivation'";
  EXPECT_NE(status.ErrorMessage().find("NotAnActivation"), std::string::npos)
      << "the error must name the activation it could not fuse, got: " << status.ErrorMessage();
}

// Every parse failure keeps its Status message, not only the unknown-name one.
TEST(Conv_WebGPU, FusedActivationParamCountMismatchIsReported) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  // LeakyRelu takes exactly one parameter.
  const Status status = InitializeFusedConv("LeakyRelu", {0.1f, 0.2f}, std::move(webgpu_ep));
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("activation_params"), std::string::npos)
      << "the error must name the attribute that is wrong, got: " << status.ErrorMessage();
}

}  // namespace test
}  // namespace onnxruntime
