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
  Ort::Env env;
  Ort::Session session_{env, "/bert_ort/leca/models/Detection/model.onnx", Ort::SessionOptions{nullptr}};

  // C API
  const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* p_env = nullptr;
  OrtLoggingLevel log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO;
  auto ret = g_ort->CreateEnv(log_level, "", &p_env);
  OrtSessionOptions* so = nullptr;
  //OrtSessionOptions so; // build error: incomplete type
  g_ort->CreateSessionOptions(&so);

  OrtSession* session;
  g_ort->CreateSession(env, "/bert_ort/leca/code/onnxruntime2/onnxruntime/test/testdata/custom_op_library/custom_op_test.onnx", so, &session);

  // demo 1
  //g_ort->RegisterCustomEPAndCustomOp("/bert_ort/leca/code/onnxruntime2/build/Linux/Debug/libtest_execution_provider.so", so);
  //void* library_path = nullptr;
  //g_ort->RegisterCustomOpsLibrary(so, "/bert_ort/leca/code/onnxruntime2/build/Linux/Debug/libcustom_op_library.so", &library_path);
  
  // demo 2
  g_ort->RegisterCustomEPAndCustomOp2("/bert_ort/leca/code/onnxruntime2/samples/customEP/build/libcustomep.so", so);
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
