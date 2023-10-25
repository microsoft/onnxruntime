#include <iostream>
#include <vector>
#include "core/session/onnxruntime_c_api.h"

int main() {
  const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* p_env = nullptr;
  OrtLoggingLevel log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO;
  auto ret = g_ort->CreateEnv(log_level, "", &p_env);
  OrtSessionOptions* so = nullptr;
  //OrtSessionOptions so; // build error: incomplete type
  g_ort->CreateSessionOptions(&so);

  std::vector<const char*> keys{"int_property", "str_property"}, values{"3", "strval"};
  g_ort->RegisterCustomEP("/bert_ort/leca/code/onnxruntime2/samples/customEP2/build/libcustomep2.so", keys.data(), values.data(), keys.size(), so);

  OrtSession* session = nullptr;
  g_ort->CreateSession(p_env, "/bert_ort/leca/models/Relu.onnx", so, &session);

  OrtMemoryInfo* memory_info = nullptr;
  g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
  float input_data[] = {-3.0f, 5.0f, -2.0f, 4.0f};
  const size_t input_len = 4 * sizeof(float);
  const int64_t input_shape[] = {4};
  const size_t shape_len = sizeof(input_shape)/sizeof(input_shape[0]);

  OrtValue* input_tensor = nullptr;
  g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_data, input_len, input_shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);

  const char* input_names[] = {"x"};
  const char* output_names[] = {"graphOut"};
  OrtValue* output_tensor = nullptr;
  g_ort->Run(session, nullptr, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor);

  float* output_tensor_data = nullptr;
  g_ort->GetTensorMutableData(output_tensor, (void**)&output_tensor_data);
  std::cout<<"Result:\n";
  for (size_t i = 0; i < 4; i++) std::cout<<output_tensor_data[i]<<" \n";

  return 0;
}

