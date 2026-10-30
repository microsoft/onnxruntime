// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "core/graph/model.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime {
namespace test {

class CudaConvDynamicShapeTest : public ::testing::TestWithParam<std::tuple<std::string, bool>> {};

TEST_P(CudaConvDynamicShapeTest, RebuildExecutionPlan) {
  const auto& [op_type, fuse_bias] = GetParam();
  OrtCUDAProviderOptionsV2 cuda_options{};
  cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
  cuda_options.fuse_conv_bias = fuse_bias;
  auto cuda_ep = CudaExecutionProviderWithOptions(&cuda_options);
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA execution provider is not available.";
  }

  Model model("cuda_conv_dynamic_shape", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 11}}, {},
              DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  ONNX_NAMESPACE::TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* shape = input_type.mutable_tensor_type()->mutable_shape();
  shape->add_dim()->set_dim_value(1);
  shape->add_dim()->set_dim_value(1);
  shape->add_dim()->set_dim_param("length");
  ONNX_NAMESPACE::TypeProto float_type;
  float_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto& x = graph.GetOrCreateNodeArg("X", &input_type);
  auto& w = graph.GetOrCreateNodeArg("W", &float_type);
  auto& b = graph.GetOrCreateNodeArg("B", &float_type);
  auto& y = graph.GetOrCreateNodeArg("Y", &input_type);
  graph.AddNode("conv", op_type, "", {&x, &w, &b}, {&y});
  ASSERT_STATUS_OK(graph.Resolve());
  std::stringstream model_stream(model.ToProto().SerializeAsString());

  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  InferenceSession session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(cuda_ep)));
  ASSERT_STATUS_OK(session.Load(model_stream));
  ASSERT_STATUS_OK(session.Initialize());

  auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  OrtValue weights, bias;
  CreateMLValue<float>(allocator, {1, 1, 1}, {2.0f}, &weights);
  CreateMLValue<float>(allocator, {1}, {3.0f}, &bias);
  const std::vector<std::string> output_names{"Y"};
  for (int64_t length : {4, 7, 4, 9}) {
    SCOPED_TRACE(length);
    std::vector<float> input(static_cast<size_t>(length));
    for (size_t i = 0; i < input.size(); ++i) {
      input[i] = static_cast<float>(i);
    }
    OrtValue input_value;
    CreateMLValue<float>(allocator, {1, 1, length}, input, &input_value);
    NameMLValMap feeds{{"X", input_value}, {"W", weights}, {"B", bias}};
    std::vector<OrtValue> fetches;
    ASSERT_STATUS_OK(session.Run(RunOptions{}, feeds, output_names, &fetches));
    ASSERT_EQ(fetches.size(), 1u);
    const auto& output = fetches[0].Get<Tensor>();
    ASSERT_EQ(output.Shape(), TensorShape({1, 1, length}));
    for (size_t i = 0; i < input.size(); ++i) {
      EXPECT_FLOAT_EQ(output.Data<float>()[i], input[i] * 2.0f + 3.0f);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    CUDA, CudaConvDynamicShapeTest,
    ::testing::Combine(::testing::Values("Conv", "ConvTranspose"), ::testing::Bool()));

}  // namespace test
}  // namespace onnxruntime
