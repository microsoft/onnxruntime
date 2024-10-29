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

const ORTCHAR_T* ep_plugin_lib = "/home/lochi/repos/ort_for_docker_ep_plugin_2/samples/tensorRTEp/build/libTensorRTEp.so"; // hardcode path for now
const ORTCHAR_T* ep_plugin_name = "tensorrtEp";
const ORTCHAR_T* model_path = "testdata/trt_ep_test_model_static_input_shape.onnx"; 
const ORTCHAR_T* model_path_2 = "testdata/trt_ep_test_model_dynamic_input_shape.onnx"; 

inline void THROW_ON_ERROR(OrtStatus* status, const OrtApi* api) {
    if (status != nullptr && api != nullptr) {
        std::cout<<"ErrorMessage:"<<api->GetErrorMessage(status)<<"\n";
        abort();
    }
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

void RunSession(Ort::Session& session,
                std::vector<const char*>& input_names,
                std::vector<Ort::Value>& ort_inputs,
                const char* const* output_names,
                std::vector<int64_t>& expected_dims,
                std::vector<float>& expected_values) {
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(), output_names, 1);
  ValidateOutputs(ort_outputs, expected_dims, expected_values);
}

void CreateSessionAndRunInference() {
  // Use C API here since EP plugin only supports C API for now
  OrtEnv* env = nullptr;
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtLoggingLevel log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR;
  THROW_ON_ERROR(api->CreateEnv(log_level, "", &env), api);
  THROW_ON_ERROR(api->RegisterPluginExecutionProviderLibrary(ep_plugin_lib, env, ep_plugin_name), api);
  OrtSessionOptions* so = nullptr;
  THROW_ON_ERROR(api->CreateSessionOptions(&so), api);
  std::vector<const char*> keys{"trt_engine_cache_enable", "trt_engine_cache_prefix", "trt_dump_ep_context_model", "trt_ep_context_file_path"};
  std::vector<const char*> values{"1", "TRTEP_Cache_Test", "1", "EP_Context_model.onnx"};
  THROW_ON_ERROR(api->SessionOptionsAppendPluginExecutionProvider(so, ep_plugin_name, env, keys.data(), values.data(), keys.size()), api);

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

/*
 * Create one session and run by multiple threads
 */
void CreateSessionAndRunInference2() {
  // Use C API
  OrtEnv* env = nullptr;
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtLoggingLevel log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR;
  THROW_ON_ERROR(api->CreateEnv(log_level, "", &env), api);
  THROW_ON_ERROR(api->RegisterPluginExecutionProviderLibrary(ep_plugin_lib, env, ep_plugin_name), api);
  OrtSessionOptions* so = nullptr;
  THROW_ON_ERROR(api->CreateSessionOptions(&so), api);
  std::vector<const char*> keys{"trt_engine_cache_enable", "trt_engine_cache_prefix", "trt_dump_ep_context_model", "trt_ep_context_file_path"};
  std::vector<const char*> values{"1", "TRTEP_Cache_Test", "1", "EP_Context_model.onnx"};
  THROW_ON_ERROR(api->SessionOptionsAppendPluginExecutionProvider(so, ep_plugin_name, env, keys.data(), values.data(), keys.size()), api);

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
  std::vector<int64_t> y_dims = {1, 3, 2};
  std::vector<float> values_y = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};

  std::vector<std::thread> threads;
  int num_thread = 5;
  for (int i = 0; i < num_thread; ++i) {
    threads.push_back(std::thread(RunSession, std::ref(session), std::ref(input_names), std::ref(ort_inputs), std::ref(output_names), std::ref(y_dims), std::ref(values_y)));
  }

  for (auto& th : threads)
    th.join();

  // Verify on cache with customized prefix
  ASSERT_TRUE(HasCacheFileWithPrefix("TRTEP_Cache_Test"));
}

TEST(TensorrtExecutionProviderPluginTest, SmallModel) {
  // Use C API
  OrtEnv* env = nullptr;
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtLoggingLevel log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR;
  THROW_ON_ERROR(api->CreateEnv(log_level, "", &env), api);
  THROW_ON_ERROR(api->RegisterPluginExecutionProviderLibrary(ep_plugin_lib, env, ep_plugin_name), api);
  OrtSessionOptions* so = nullptr;
  THROW_ON_ERROR(api->CreateSessionOptions(&so), api);
  std::vector<const char*> keys;
  std::vector<const char*> values;
  THROW_ON_ERROR(api->SessionOptionsAppendPluginExecutionProvider(so, ep_plugin_name, env, keys.data(), values.data(), keys.size()), api);

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
    threads.push_back(std::thread(CreateSessionAndRunInference));

  for (auto& th : threads)
    th.join();
}

TEST(TensorrtExecutionProviderPluginTest, SessionCreationWithSingleThreadAndInferenceWithMultiThreads) {
  std::vector<int> dims = {1, 3, 2};

  CreateSessionAndRunInference2();
}

TEST(TensorrtExecutionProviderPluginTest, EPContextNode) {
  // Use C API
  OrtEnv* env = nullptr;
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtLoggingLevel log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR;
  THROW_ON_ERROR(api->CreateEnv(log_level, "", &env), api);
  THROW_ON_ERROR(api->RegisterPluginExecutionProviderLibrary(ep_plugin_lib, env, ep_plugin_name), api);

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
  std::vector<int64_t> y_dims = {1, 3, 2};
  std::vector<float> values_y = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};

  /*
   * Test case 1: Dump context model
   *
   * provider options=>
   *   trt_ep_context_file_path = "EP_Context_model.onnx"
   *
   * expected result =>
   *   context model "EP_Context_model.onnx" should be created in current directory
   *
   */
  OrtSessionOptions* so = nullptr;
  THROW_ON_ERROR(api->CreateSessionOptions(&so), api);
  std::vector<const char*> keys{"trt_engine_cache_enable", "trt_dump_ep_context_model", "trt_ep_context_file_path"};
  std::vector<const char*> values{"1", "1", "EP_Context_model.onnx"};
  THROW_ON_ERROR(api->SessionOptionsAppendPluginExecutionProvider(so, ep_plugin_name, env, keys.data(), values.data(), keys.size()), api);

  Ort::SessionOptions ort_so{so};
  Ort::Env ort_env{env};
  Ort::Session session(ort_env, model_path, ort_so);

  ASSERT_TRUE(HasCacheFileWithPrefix("EP_Context_model.onnx"));

  /*
   * Test case 2: Dump context model
   *
   * provider options=>
   *   trt_engine_cache_prefix = "TRT_engine_cache"
   *   trt_ep_context_file_path = "context_model_folder"
   *   trt_engine_cache_path = "engine_cache_folder"
   *
   * expected result =>
   *   engine cache "./context_model_folder/engine_cache_folder/TRT_engine_cache...engine" should be created
   *   context model "./context_model_folder/trt_ep_test_model_static_input_shape_ctx.onnx" should be created
   */
  OrtSessionOptions* so2 = nullptr;
  THROW_ON_ERROR(api->CreateSessionOptions(&so2), api);
  std::vector<const char*> keys2{"trt_engine_cache_enable", "trt_dump_ep_context_model", "trt_engine_cache_prefix", "trt_engine_cache_path", "trt_ep_context_file_path"};
  std::vector<const char*> values2{"1", "1", "TRT_engine_cache", "engine_cache_folder", "context_model_folder"};
  THROW_ON_ERROR(api->SessionOptionsAppendPluginExecutionProvider(so2, ep_plugin_name, env, keys2.data(), values2.data(), keys2.size()), api);

  Ort::SessionOptions ort_so2{so2};
  Ort::Session session2(ort_env, model_path, ort_so2);

  auto new_engine_cache_path = std::filesystem::path("context_model_folder").append("engine_cache_folder").string();
  // Test engine cache path:
  // "./context_model_folder/engine_cache_folder/TRT_engine_cache...engine" should be created
  ASSERT_TRUE(HasCacheFileWithPrefix("TRT_engine_cache", new_engine_cache_path));
  // Test context model path:
  // "./context_model_folder/trt_ep_test_model_static_input_shape_ctx.onnx" should be created
  ASSERT_TRUE(HasCacheFileWithPrefix("trt_ep_test_model_static_input_shape_ctx.onnx", "context_model_folder"));

  /*
   * Test case 3: Run the dumped context model
   *
   * context model path = "./EP_Context_model.onnx" (created from case 1)
   *
   * expected result=>
   *   engine cache is also in the same current dirctory as "./xxxxx.engine"
   *   and the "ep_cache_context" attribute node of the context model should point to that.
   *
   */
  OrtSessionOptions* so3 = nullptr;
  THROW_ON_ERROR(api->CreateSessionOptions(&so3), api);
  std::vector<const char*> keys3{"trt_engine_cache_enable"};
  std::vector<const char*> values3{"1"};
  THROW_ON_ERROR(api->SessionOptionsAppendPluginExecutionProvider(so3, ep_plugin_name, env, keys3.data(), values3.data(), keys3.size()), api);

  Ort::SessionOptions ort_so3{so3};
  Ort::Session session3(ort_env, "EP_Context_model.onnx", ort_so3);

  // Run inference
  auto ort_outputs3 = session3.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                 output_names, 1);
  // Validate results
  ValidateOutputs(ort_outputs3, y_dims, values_y);
}

}  // namespace onnxruntime
