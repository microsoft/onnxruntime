// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/inference_session.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test_allocator.h"
#include "gtest/gtest.h"

// defined in test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

TEST(AzureEP, TestSessionCreation) {
  const auto* ort_model_path = ORT_TSTR("testdata/mul_1.onnx");
  Ort::SessionOptions so;
  so.AddConfigEntry("azure.endpoint_type", "triton");
  onnxruntime::ProviderOptions options;
  so.AppendExecutionProvider("AZURE", options);
  //session could be created
  EXPECT_NO_THROW((Ort::Session{*ort_env, ort_model_path, so}));
}

TEST(AzureEP, TestSessionRunMissingConfig) {
  const auto* ort_model_path = ORT_TSTR("testdata/mul_1.onnx");
  Ort::SessionOptions so;
  onnxruntime::ProviderOptions options;
  so.AppendExecutionProvider("AZURE", options);
  Ort::Session sess(*ort_env, ort_model_path, so);

  float raw_inputs[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  std::vector<int64_t> input_dims = {3, 2};
  std::vector<Ort::Value> input_values;
  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y"};
  auto default_allocator = std::make_unique<MockedOrtAllocator>();
  input_values.emplace_back(Ort::Value::CreateTensor<float>(default_allocator->Info(), raw_inputs, 6, input_dims.data(), 2));

  Ort::RunOptions run_options;
  run_options.AddConfigEntry("use_azure", "1");
  //exception expected due to the missing of endpoint type
  EXPECT_THROW(sess.Run(run_options, input_names, input_values.data(), 1UL, output_names, 1UL), Ort::Exception);
}

TEST(AzureEP, TestSessionRunMissingEP) {
  const auto* ort_model_path = ORT_TSTR("testdata/mul_1.onnx");
  Ort::SessionOptions so;
  so.AddConfigEntry("azure.endpoint_type", "triton");
  Ort::Session sess(*ort_env, ort_model_path, so);

  float raw_inputs[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  std::vector<int64_t> input_dims = {3, 2};
  std::vector<Ort::Value> input_values;
  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y"};
  auto default_allocator = std::make_unique<MockedOrtAllocator>();
  input_values.emplace_back(Ort::Value::CreateTensor<float>(default_allocator->Info(), raw_inputs, 6, input_dims.data(), 2));

  Ort::RunOptions run_options;
  //local inference should work
  EXPECT_NO_THROW(sess.Run(run_options, input_names, input_values.data(), 1UL, output_names, 1UL));

  run_options.AddConfigEntry("use_azure", "1");
  //exception expected due to the missing of azure.uri and azure.model_name
  EXPECT_THROW(sess.Run(run_options, input_names, input_values.data(), 1UL, output_names, 1UL), Ort::Exception);
}

TEST(AzureEP, TestSessionRunWrongUri) {
  const auto* ort_model_path = ORT_TSTR("testdata/mul_1.onnx");
  Ort::SessionOptions so;
  so.AddConfigEntry("azure.endpoint_type", "triton");
  so.AddConfigEntry("azure.uri", "https://12341nbasdkjah1239045dsknldjnlvqwf.com");
  so.AddConfigEntry("azure.model_name", "modelnotexist");
  Ort::Session sess(*ort_env, ort_model_path, so);

  float raw_inputs[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  std::vector<int64_t> input_dims = {3, 2};
  std::vector<Ort::Value> input_values;
  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y"};
  auto default_allocator = std::make_unique<MockedOrtAllocator>();
  input_values.emplace_back(Ort::Value::CreateTensor<float>(default_allocator->Info(), raw_inputs, 6, input_dims.data(), 2));

  Ort::RunOptions run_options;
  run_options.AddConfigEntry("use_azure", "1");
  run_options.AddConfigEntry("azure.auth_key", "asdjfakldkvnlkajefoiauh32hriunive2324");
  //exception expected due to the non-existing endpoint
  EXPECT_THROW(sess.Run(run_options, input_names, input_values.data(), 1UL, output_names, 1UL), Ort::Exception);
}

}  // namespace test
}  // namespace onnxruntime