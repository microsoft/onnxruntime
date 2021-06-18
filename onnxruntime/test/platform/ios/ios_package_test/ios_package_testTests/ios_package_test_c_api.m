// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
//  ios_package_test_c_api.m
//  ios_package_testTests
//
//  This file hosts the tests of ORT C API, for tests of ORT C++ API, please see ios_package_test_cpp_api.mm
//

#import <XCTest/XCTest.h>
#include <math.h>
#include <onnxruntime/onnxruntime_c_api.h>

#define ASSERT_ON_ERROR(expr)                                      \
  do {                                                             \
    OrtStatus* status = (expr);                                    \
    XCTAssertEqual(NULL, status, @"Failed with error message: %@", \
                   @(ort_env_->GetErrorMessage(status)));          \
  } while (0)

@interface ios_package_test_c_api : XCTestCase {
  const OrtApi* ort_env_;
}

@end

@implementation ios_package_test_c_api

- (void)setUp {
  // Put setup code here. This method is called before the invocation of each test method in the class.
  ort_env_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
}

- (void)tearDown {
  // Put teardown code here. This method is called after the invocation of each test method in the class.
}

- (void)testCAPI {
  // This is an e2e test for ORT C API
  OrtEnv* env = NULL;
  ASSERT_ON_ERROR(ort_env_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "testCAPI", &env));

  // initialize session options if needed
  OrtSessionOptions* session_options;
  ASSERT_ON_ERROR(ort_env_->CreateSessionOptions(&session_options));
  ASSERT_ON_ERROR(ort_env_->SetIntraOpNumThreads(session_options, 1));

  OrtSession* session;
  NSString* ns_model_path = [[NSBundle mainBundle] pathForResource:@"sigmoid" ofType:@"ort"];
  ASSERT_ON_ERROR(ort_env_->CreateSession(env, ns_model_path.UTF8String, session_options, &session));

  size_t input_tensor_size = 3 * 4 * 5;
  float input_tensor_values[input_tensor_size];
  float expected_output_values[input_tensor_size];
  const char* input_node_names[] = {"x"};
  const char* output_node_names[] = {"y"};
  const int64_t input_node_dims[] = {3, 4, 5};

  for (size_t i = 0; i < input_tensor_size; i++) {
    input_tensor_values[i] = (float)i - 30;
    expected_output_values[i] = 1.0f / (1 + exp(-input_tensor_values[i]));
  }

  OrtMemoryInfo* memory_info;
  ASSERT_ON_ERROR(ort_env_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  OrtValue* input_tensor = NULL;
  ASSERT_ON_ERROR(ort_env_->CreateTensorWithDataAsOrtValue(
      memory_info, input_tensor_values, input_tensor_size * sizeof(float),
      input_node_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
  int is_tensor;
  ASSERT_ON_ERROR(ort_env_->IsTensor(input_tensor, &is_tensor));
  XCTAssertNotEqual(is_tensor, 0);
  ort_env_->ReleaseMemoryInfo(memory_info);

  OrtValue* output_tensor = NULL;
  ASSERT_ON_ERROR(ort_env_->Run(session, NULL, input_node_names,
                                (const OrtValue* const*)&input_tensor, 1,
                                output_node_names, 1, &output_tensor));
  ASSERT_ON_ERROR(ort_env_->IsTensor(output_tensor, &is_tensor));
  XCTAssertNotEqual(is_tensor, 0);

  // Get pointer to output tensor float values
  float* output_values;
  ASSERT_ON_ERROR(ort_env_->GetTensorMutableData(output_tensor, (void**)&output_values));

  for (size_t i = 0; i < input_tensor_size; i++) {
    XCTAssertEqualWithAccuracy(expected_output_values[i], output_values[i], 1e-6);
  }

  ort_env_->ReleaseValue(output_tensor);
  ort_env_->ReleaseValue(input_tensor);
  ort_env_->ReleaseSession(session);
  ort_env_->ReleaseSessionOptions(session_options);
  ort_env_->ReleaseEnv(env);
}

@end
