// ConsoleApplicationUsingORT.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include "core/framework/tensor_shape2.h"
//#include "MathFunctions.h"
#include "core/common/code_location.h"
#include "core/common/denormal.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"

int main()
{
  std::vector<int64_t> dims{3, 4, 5};
  //onnxruntime::TensorShape2 tensor_shape_2(dims);
  //onnxruntime::TensorShape2 tensor_shape_2;
  //std::string str = tensor_shape_2.ToString();
  //std::cout << "Hello World! " << str << "\n ";
  std::cout<<"hello dims size="<<dims.size()<<"\n";
//  std::cout<<"sqrt of 16:"<<mathfunctions::sqrt(16.0)<<"\n";
  //std::cout<<"cube:"<<onnxruntime::cube(3)<<"\n";
  std::cout<<"cube:"<<onnxruntime::inlineCube(3)<<"\n";

  onnxruntime::CodeLocation loc("file", 3, "func", {});
  std::cout<<"loc.string:"<<loc.ToString()<<"\n";

  std::cout<<"SetDenormalAsZero:"<<onnxruntime::SetDenormalAsZero(true)<<"\n";	// build error: undefined reference to

  // C++ API
//  Ort::Env env;
//  Ort::Session session_{env, "/bert_ort/leca/models/Detection/model.onnx", Ort::SessionOptions{nullptr}};

  // C API
  const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* p_env = nullptr;
  OrtLoggingLevel log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO;
  auto ret = g_ort->CreateEnv(log_level, "", &p_env);
  OrtSessionOptions* so = nullptr;
  //OrtSessionOptions so; // build error: incomplete type
  g_ort->CreateSessionOptions(&so);

  // demo 1
  //g_ort->RegisterCustomEPAndCustomOp("/bert_ort/leca/code/onnxruntime2/build/Linux/Debug/libtest_execution_provider.so", so);
  //void* library_path = nullptr;
  //g_ort->RegisterCustomOpsLibrary(so, "/bert_ort/leca/code/onnxruntime2/build/Linux/Debug/libcustom_op_library.so", &library_path);

  // demo 2
  g_ort->RegisterCustomEPAndCustomOp2("/bert_ort/leca/code/onnxruntime2/samples/customEP/build/libcustomep.so", so);

  OrtSession* session = nullptr;
  g_ort->CreateSession(p_env, "/bert_ort/leca/models/CustomOpTwo.onnx", so, &session);

  OrtMemoryInfo* memory_info = nullptr;
  g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
  float input_data[] = {0.4f, 1.5f, 2.6f, 3.7f};
  const size_t input_len = 4 * sizeof(float);
  const int64_t input_shape[] = {4};
  const size_t shape_len = sizeof(input_shape)/sizeof(input_shape[0]);

  OrtValue* input_tensor = nullptr;
  g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_data, input_len, input_shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);

  const char* input_names[] = {"x"};
  const char* output_names[] = {"graphOut"};
  OrtValue* output_tensor;
  g_ort->Run(session, nullptr, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor);
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started:
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
