// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/session/onnxruntime_cxx_api.h"
#include "test/testdata/custom_op_library/custom_op_library.h"

#ifdef _WIN32
typedef const wchar_t* PATH_TYPE;
#define TSTR(X) L##X
#else
#define TSTR(X) (X)
typedef const char* PATH_TYPE;
#endif

extern std::unique_ptr<Ort::Env> ort_env;

// Test OrtApi RegisterCustomOps.
// Replicate the expected mobile setup where the binary is linked against onnxruntime and a custom ops library.
// In the test we use testdata/custom_op_library. In mobile scenarios onnxruntime-extensions would provide custom ops.
TEST(CustomOpRegistration, TestUsingFuncName) {
  Ort::SessionOptions session_options;

  void* addr = RegisterCustomOps;
  std::cout << "RegisterCustomOps addr " << addr << "\n";

  // RegisterUnitTestCustomOps will add the Foo custom op in domain ort_unit_test
  // custom_op_library has RegisterCustomOps and RegisterCustomOpsAltName as options for registration functions.
  // Call RegisterCustomOpsAltName to test the path which does not use the default name.
  Ort::ThrowOnError(Ort::GetApi().RegisterCustomOps(session_options, "RegisterCustomOpsAltName"));

  // load model containing nodes using the custom op/s to validate. will throw if custom op wasn't registered.
  static constexpr PATH_TYPE model = TSTR("testdata/custom_op_library/custom_op_test.onnx");
  Ort::Session session(*ort_env, model, session_options);
}
