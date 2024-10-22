// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "gtest/gtest.h"

#include <iostream>
#include <string>
#include <thread>
#include <filesystem>
#include <chrono>

namespace onnxruntime {

const ORTCHAR_T* ep_plugin_lib = "/home/lochi/repos/ort_for_docker_ep_plugin/samples/tensorRTEp/build/libTensorRTEp.so"; // hardcode path for now
const ORTCHAR_T* ep_plugin_name = "tensorrtEp";
const ORTCHAR_T* model_path = "testdata/trt_ep_test_model_static_input_shape.onnx"; 
const ORTCHAR_T* model_path_2 = "testdata/trt_ep_test_model_dynamic_input_shape.onnx"; 

inline void THROW_ON_ERROR(OrtStatus* status, const OrtApi* api) {
    if (status != nullptr && api != nullptr) {
        std::cout<<"ErrorMessage:"<<api->GetErrorMessage(status)<<"\n";
        abort();
    }
}

void RegisterTrtEpPlugin(const OrtApi* api, OrtEnv* env, OrtSessionOptions* so, std::vector<const char*>& keys, std::vector<const char*>& values) {
    std::cout << keys.size() << std::endl;
    THROW_ON_ERROR(api->RegisterPluginExecutionProviderLibrary(ep_plugin_lib, env, ep_plugin_name), api);
    THROW_ON_ERROR(api->SessionOptionsAppendPluginExecutionProvider(so, ep_plugin_name, env, keys.data(), values.data(), keys.size()), api);
}

bool HasCacheFileWithPrefix(const std::string& prefix, std::string file_dir = "") {
  std::filesystem::path target_dir;
  if (file_dir.empty()) {
    target_dir = std::filesystem::current_path();
  } else {
    target_dir = std::filesystem::path(file_dir);
  }

  for (const auto& entry : std::filesystem::directory_iterator(target_dir)) {
    if (entry.is_regular_file()) {
      std::string filename = entry.path().filename().string();
      if (filename.rfind(prefix, 0) == 0) {
        return true;
      }
    }
  }
  return false;
}

void ValidateOutputs(std::vector<Ort::Value>& ort_outputs,
                std::vector<int64_t>& expected_dims,
                std::vector<float>& expected_values) {

  auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), expected_dims);
  size_t total_len = type_info.GetElementCount();
  ASSERT_EQ(expected_values.size(), total_len);

  float* f = ort_outputs[0].GetTensorMutableData<float>();
  for (size_t i = 0; i != total_len; ++i) {
    ASSERT_EQ(expected_values[i], f[i]);
  }
}

void RunWithOneSessionSingleThreadInference() {
  // Use C API at first since EP plugin only supports C API for now
  OrtEnv* env = nullptr;
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtLoggingLevel log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR;
  THROW_ON_ERROR(api->CreateEnv(log_level, "", &env), api);
  OrtSessionOptions* so = nullptr;
  THROW_ON_ERROR(api->CreateSessionOptions(&so), api);

  std::vector<const char*> keys{"trt_engine_cache_enable", "trt_engine_cache_prefix", "trt_dump_ep_context_model", "trt_ep_context_file_path"};
  std::vector<const char*> values{"1", "TRTEP_Cache_Test", "1", "EP_Context_model.onnx"};

  RegisterTrtEpPlugin(api, env, so, keys, values);

  // Use C++ Wrapper
  Ort::SessionOptions ort_so{so};
  Ort::Env ort_env{env};


  Ort::Session session(ort_env, model_path, ort_so);

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  // input 0, 1, 2
  std::vector<float> input_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector<int64_t> input_dims = {1, 3, 2};
  input_names.emplace_back("X");
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_data.data()),
                                      input_data.size(), input_dims.data(), input_dims.size()));
  input_names.emplace_back("Y");
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_data.data()),
                                      input_data.size(), input_dims.data(), input_dims.size()));
  input_names.emplace_back("Z");
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_data.data()),
                                      input_data.size(), input_dims.data(), input_dims.size()));

  // output 0
  const char* output_names[] = {"M"};

  // Run inference
  // TRT engine will be created and cached
  // TRT profile will be created and cached only for dynamic input shape
  // Data in profile,
  // X: 1, 3, 3, 2, 2, 2
  // Y: 1, 3, 3, 2, 2, 2
  // Z: 1, 3, 3, 2, 2, 2
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                 output_names, 1);

  // Verify on cache with customized prefix
  ASSERT_TRUE(HasCacheFileWithPrefix("TRTEP_Cache_Test"));

  // Verify EP context model with user provided name
  ASSERT_TRUE(HasCacheFileWithPrefix("EP_Context_model.onnx"));
}

TEST(TensorrtExecutionProviderPluginTest, SmallModel) {
  // Use C API at first since EP plugin only supports C API for now
  OrtEnv* env = nullptr;
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtLoggingLevel log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR;
  THROW_ON_ERROR(api->CreateEnv(log_level, "", &env), api);
  OrtSessionOptions* so = nullptr;
  THROW_ON_ERROR(api->CreateSessionOptions(&so), api);

  std::vector<const char*> keys;
  std::vector<const char*> values;

  RegisterTrtEpPlugin(api, env, so, keys, values);

  // Use C++ Wrapper
  Ort::SessionOptions ort_so{so};
  Ort::Env ort_env{env};
  Ort::Session session(ort_env, model_path, ort_so);

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  // input 0, 1, 2
  std::vector<float> input_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector<int64_t> input_dims = {1, 3, 2};
  input_names.emplace_back("X");
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_data.data()),
                                      input_data.size(), input_dims.data(), input_dims.size()));
  input_names.emplace_back("Y");
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_data.data()),
                                      input_data.size(), input_dims.data(), input_dims.size()));
  input_names.emplace_back("Z");
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_data.data()),
                                      input_data.size(), input_dims.data(), input_dims.size()));

  // output 0
  const char* output_names[] = {"M"};

  // Run inference
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                 output_names, 1);

  // Validate results
  std::vector<int64_t> y_dims = {1, 3, 2};
  std::vector<float> values_y = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};
  ValidateOutputs(ort_outputs, y_dims, values_y);
}

TEST(TensorrtExecutionProviderPluginTest, SessionCreationWithMultiThreadsAndInferenceWithMultiThreads) {
  std::vector<std::thread> threads;
  std::vector<int> dims = {1, 3, 2};
  int num_thread = 1;

  for (int i = 0; i < num_thread; ++i)
    threads.push_back(std::thread(RunWithOneSessionSingleThreadInference));

  for (auto& th : threads)
    th.join();
}

}  // namespace onnxruntime
