// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// ONNX models aren't supported in a minimal build, and the custom ops need additional infrastructure to test if CUDA
// is enabled. As we're really testing the symbol lookup, keep it simple and skip if USE_CUDA is defined.
#if !defined(ORT_MINIMAL_BUILD) && !defined(USE_CUDA)

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
static constexpr PATH_TYPE TestModel = TSTR("testdata/custom_op_library/custom_op_test.onnx");
#if !defined(DISABLE_FLOAT8_TYPES)
static constexpr PATH_TYPE TestModelFloat8 = TSTR("testdata/custom_op_library/custom_op_test_float8.onnx");
#endif

// Test OrtApi RegisterCustomOpsUsingFunction.
// Replicate the expected mobile setup where the binary is linked against onnxruntime and a custom ops library.
// In the test we use testdata/custom_op_library. In mobile scenarios onnxruntime-extensions would provide custom ops.
TEST(CustomOpRegistration, TestUsingFuncNameCApi) {
  Ort::SessionOptions session_options;

  // need to reference something in the custom ops library to prevent it being thrown away by the linker
  void* addr = reinterpret_cast<void*>(RegisterCustomOpsAltName);
  std::cout << "RegisterCustomOpsAltName addr " << addr << "\n";

  // RegisterUnitTestCustomOps will add the Foo custom op in domain ort_unit_test
  // custom_op_library has RegisterCustomOps and RegisterCustomOpsAltName as options for registration functions.
  // Call RegisterCustomOpsAltName to test the path which does not use the default name.
  Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsUsingFunction(session_options, "RegisterCustomOpsAltName"));

  // load model containing nodes using the custom op/s to validate. will throw if custom op wasn't registered.
  Ort::Session session(*ort_env, TestModel, session_options);
}

#if !defined(DISABLE_FLOAT8_TYPES)

TEST(CustomOpRegistration, TestUsingFuncNameCApiFloat8) {
  // Test similar to TestUsingFuncNameCApi but loads model custom_op_test_float8.onnx
  // which uses type Float8E4M3FN.
  Ort::SessionOptions session_options;
  session_options.RegisterCustomOpsUsingFunction("RegisterCustomOpsAltName");
  Ort::Session session(*ort_env, TestModelFloat8, session_options);
}

#endif

TEST(CustomOpRegistration, TestUsingFuncNameCxxApi) {
  Ort::SessionOptions session_options;
  session_options.RegisterCustomOpsUsingFunction("RegisterCustomOpsAltName");
  Ort::Session session(*ort_env, TestModel, session_options);
}

#endif  // !defined(ORT_MINIMAL_BUILD)
