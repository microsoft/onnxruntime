// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include "gtest/gtest.h"

#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "test/test_environment.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

// ConvActivationFusion writes the attribute from the core library while the activation set lives
// in the EP, so a core library newer than the EP binary can fuse an activation the kernel has
// never heard of. That has to report the activation by name, not a bare failed assertion.
TEST(Conv_WebGPU, UnknownFusedActivationIsReportedByName) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

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
  conv.AddAttribute("activation", "NotAnActivation");

  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  SessionOptions so;
  so.session_logid = "Conv_WebGPU.UnknownFusedActivationIsReportedByName";
  InferenceSession session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(webgpu_ep)));
  ASSERT_STATUS_OK(session.Load(model_data.data(), static_cast<int>(model_data.size())));

  const Status status = session.Initialize();
  ASSERT_FALSE(status.IsOK()) << "the WebGPU EP has no shader for 'NotAnActivation'";
  EXPECT_NE(status.ErrorMessage().find("NotAnActivation"), std::string::npos)
      << "the error must name the activation it could not fuse, got: " << status.ErrorMessage();
}

}  // namespace test
}  // namespace onnxruntime
