#include <iostream>
#include <string>
#include <unordered_map>
#include "core/session/onnxruntime_c_api.h"

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

inline void THROW_ON_ERROR(OrtStatus* status) {
  if (status != nullptr) {
    std::cout << "ErrorMessage:" << g_ort->GetErrorMessage(status) << "\n";
    abort();
  }
}

void RunRelu(const OrtApi* g_ort, OrtEnv* p_env, OrtSessionOptions* so) {
  OrtSession* session = nullptr;
  // Copy relu.onnx model from winml\test\collateral\models to the same path as the executable
  THROW_ON_ERROR(g_ort->CreateSession(p_env, L"relu.onnx", so, &session));

  OrtMemoryInfo* memory_info = nullptr;
  THROW_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  float input_data[] = {-3.0f, 5.0f, -2.0f, 4.0f, 0.0f};
  const size_t input_len = 5 * sizeof(float);
  const int64_t input_shape[] = {5};
  const size_t shape_len = sizeof(input_shape) / sizeof(input_shape[0]);

  OrtValue* input_tensor = nullptr;
  THROW_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_data, input_len, input_shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y"};
  OrtValue* output_tensor = nullptr;
  THROW_ON_ERROR(g_ort->Run(session, nullptr, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor));

  float* output_tensor_data = nullptr;
  THROW_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_tensor_data));
  std::cout << "Result:\n";
  for (size_t i = 0; i < 5; i++) std::cout << output_tensor_data[i] << " \n";
}

int main() {
  int a;
  std::cout << "prepare to attach:";
  std::cin >> a;

  OrtEnv* p_env = nullptr;
  OrtLoggingLevel log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR;  // OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO;
  THROW_ON_ERROR(g_ort->CreateEnv(log_level, "", &p_env));
  OrtSessionOptions* so = nullptr;
  THROW_ON_ERROR(g_ort->CreateSessionOptions(&so));

  OrtTensorRTProviderOptionsV2* tensorrt_options = nullptr;
  THROW_ON_ERROR(g_ort->CreateTensorRTProviderOptions(&tensorrt_options));
  THROW_ON_ERROR(g_ort->SessionOptionsAppendExecutionProvider_TensorRT_V2(so, tensorrt_options));

  std::unordered_map<std::string, std::string> ov_options;
  ov_options["device_type"] = "CPU";
  ov_options["precision"] = "FP32";
  std::vector<const char*> keys, values;
  for (const auto& entry : ov_options) {
    keys.push_back(entry.first.c_str());
    values.push_back(entry.second.c_str());
  }
  THROW_ON_ERROR(g_ort->SessionOptionsAppendExecutionProvider_OpenVINO_V2(so, keys.data(), values.data(), keys.size()));

  RunRelu(g_ort, p_env, so);

  return 0;
}
