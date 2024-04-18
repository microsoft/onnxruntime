// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#define ORT_MODEL_FOLDER ORT_TSTR("testdata/")

// in test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
TEST(CPUExecutionProviderTest, MetadataTest) {
  CPUExecutionProviderInfo info;
  auto provider = std::make_unique<CPUExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);
  ASSERT_EQ(provider->GetOrtDeviceByMemType(OrtMemTypeDefault).Type(), OrtDevice::CPU);
}

// TODO: Remove. This is a throwaway test for Int4
TEST(CPUExecutionProviderTest, Example_Conv_Int4) {
  Ort::SessionOptions so;

  // Ensure all type/shape inference warnings result in errors!
  so.AddConfigEntry(kOrtSessionOptionsConfigStrictShapeTypeInference, "1");
  so.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
  const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "conv.int4.int8.qdq.onnx";
  Ort::Session session(*ort_env, ort_model_path, so);

  std::array<float, 1 * 3 * 8 * 8> input0_data = {};
  for (size_t i = 0; i < input0_data.size(); i++) {
    input0_data[i] = 0.2f;
  }

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> ort_input_names;

  // Add input0
  std::array<int64_t, 4> inputs_shape{1, 3, 8, 8};
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, input0_data.data(), input0_data.size(), inputs_shape.data(), inputs_shape.size()));
  ort_input_names.push_back("input_0");

  // Run session and get outputs
  std::array<const char*, 1> output_names{"output_0"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, ort_input_names.data(), ort_inputs.data(),
                                                    ort_inputs.size(), output_names.data(), output_names.size());

  // Check output shape.
  Ort::Value& ort_output = ort_outputs[0];
  auto typeshape = ort_output.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> output_shape = typeshape.GetShape();

  EXPECT_THAT(output_shape, ::testing::ElementsAre(1, 5, 6, 6));
  const float* results = ort_output.GetTensorData<float>();

  for (size_t i = 0; i < typeshape.GetElementCount(); i++) {
    std::cout << i << ": " << results[i] << std::endl;
  }
}
}  // namespace test
}  // namespace onnxruntime
