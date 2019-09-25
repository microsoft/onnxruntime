// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
#include "CppUnitTest.h"
#include <assert.h>
#include <onnxruntime_c_api.h>

wchar_t* GetWideString(const char* c) {
  const size_t cSize = strlen(c) + 1;
  wchar_t* wc = new wchar_t[cSize];
  mbstowcs(wc, c, cSize);

  return wc;
}

#define ORT_ABORT_ON_ERROR(expr)                         \
  {                                                      \
    OrtStatus* onnx_status = (expr);                     \
    if (onnx_status != NULL) {                           \
      const char* msg = OrtGetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                      \
      OrtReleaseStatus(onnx_status);                     \
      wchar_t* wmsg = GetWideString(msg);                \
      Assert::Fail(L"Failed on ORT_ABORT_ON_ERROR");     \
      free(wmsg);                                        \
    }                                                    \
  }

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTest1 {
TEST_CLASS(UnitTest1){
  public :

      int run_inference(OrtSession * session){
          size_t input_height = 224;
size_t input_width = 224;
float* model_input = (float*)malloc(sizeof(float) * 224 * 224 * 3);
size_t model_input_ele_count = 224 * 224 * 3;

// initialize to values between 0.0 and 1.0
for (unsigned int i = 0; i < model_input_ele_count; i++)
  model_input[i] = (float)i / (float)(model_input_ele_count + 1);

OrtMemoryInfo* memory_info;
ORT_ABORT_ON_ERROR(OrtCreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
const size_t input_shape[] = {1, 3, 224, 224};
const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
const size_t model_input_len = model_input_ele_count * sizeof(float);

OrtValue* input_tensor = NULL;
ORT_ABORT_ON_ERROR(OrtCreateTensorWithDataAsOrtValue(memory_info, model_input, model_input_len, input_shape, input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
assert(input_tensor != NULL);
assert(OrtIsTensor(input_tensor));
OrtReleaseMemoryInfo(memory_info);
const char* input_names[] = {"data_0"};
const char* output_names[] = {"softmaxout_1"};
OrtValue* output_tensor = NULL;
ORT_ABORT_ON_ERROR(OrtRun(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor));
assert(output_tensor != NULL);
assert(OrtIsTensor(output_tensor));

OrtReleaseValue(output_tensor);
OrtReleaseValue(input_tensor);
free(model_input);
return 0;
}  // namespace UnitTest1

int test() {
  const wchar_t* model_path = L"squeezenet.onnx";
  OrtEnv* env;
  ORT_ABORT_ON_ERROR(OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
  OrtSessionOptions* session_option = OrtCreateSessionOptions();
  OrtSession* session;
  OrtSetIntraOpNumThreads(session_option, 1);
  ORT_ABORT_ON_ERROR(OrtCreateSession(env, model_path, session_option, &session));
  OrtReleaseSessionOptions(session_option);

  int result = run_inference(session);

  OrtReleaseSession(session);
  OrtReleaseEnv(env);
}

TEST_METHOD(TestMethod1) {
  int res = test();
  Assert::AreEqual(res, 0);
}
}
;
}
