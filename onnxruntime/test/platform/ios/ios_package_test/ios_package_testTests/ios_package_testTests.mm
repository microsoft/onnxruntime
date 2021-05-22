// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
//  ios_package_testTests.m
//  ios_package_testTests
//

#import <XCTest/XCTest.h>
#include <math.h>
#include <onnxruntime/onnxruntime_c_api.h>
#include <onnxruntime/onnxruntime_cxx_api.h>

#define ASSERT_ON_ERROR(expr)                                                                               \
  do {                                                                                                      \
    OrtStatus* status = (expr);                                                                             \
    XCTAssertEqual(nullptr, status, @"Failed with error message: %@", @(ort_api->GetErrorMessage(status))); \
  } while (0);

@interface ios_package_testTests : XCTestCase

@end

@implementation ios_package_testTests

- (void)setUp {
  // Put setup code here. This method is called before the invocation of each test method in the class.
}

- (void)tearDown {
  // Put teardown code here. This method is called after the invocation of each test method in the class.
}

- (void)testCAPI {
  // This is an e2e test for ORT C API
  const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* env = NULL;
  ASSERT_ON_ERROR(ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "testCAPI", &env));

  // initialize session options if needed
  OrtSessionOptions* so;
  ASSERT_ON_ERROR(ort_api->CreateSessionOptions(&so));

  OrtSession* session;
  NSString* path = [[NSBundle mainBundle] pathForResource:@"sigmoid" ofType:@"ort"];
  const char* cPath = [path cStringUsingEncoding:NSUTF8StringEncoding];
  ASSERT_ON_ERROR(ort_api->CreateSession(env, cPath, so, &session));

  OrtAllocator* allocator;
  ASSERT_ON_ERROR(ort_api->GetAllocatorWithDefaultOptions(&allocator));

  size_t input_tensor_size = 3 * 4 * 5;
  float input_tensor_values[input_tensor_size];
  float expected_values[input_tensor_size];
  const char* input_node_names[] = {"x"};
  const char* output_node_names[] = {"y"};
  const int64_t input_node_dims[] = {3, 4, 5};

  for (size_t i = 0; i < input_tensor_size; i++) {
    input_tensor_values[i] = (float)i - 30;
    expected_values[i] = 1.0f / (1 + exp(-input_tensor_values[i]));
  }

  OrtMemoryInfo* memory_info;
  ASSERT_ON_ERROR(ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  OrtValue* input_tensor = NULL;
  ASSERT_ON_ERROR(ort_api->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values, input_tensor_size * sizeof(float), input_node_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
  int is_tensor;
  ASSERT_ON_ERROR(ort_api->IsTensor(input_tensor, &is_tensor));
  XCTAssertNotEqual(is_tensor, 0);
  ort_api->ReleaseMemoryInfo(memory_info);

  OrtValue* output_tensor = NULL;
  ASSERT_ON_ERROR(ort_api->Run(session, NULL, input_node_names, (const OrtValue* const*)&input_tensor, 1, output_node_names, 1, &output_tensor));
  ASSERT_ON_ERROR(ort_api->IsTensor(output_tensor, &is_tensor));
  XCTAssertNotEqual(is_tensor, 0);

  // Get pointer to output tensor float values
  float* output_values;
  ASSERT_ON_ERROR(ort_api->GetTensorMutableData(output_tensor, (void**)&output_values));

  for (size_t i = 0; i < input_tensor_size; i++) {
    NSLog(@"%1.10f\t%1.10f", expected_values[i], output_values[i]);
    XCTAssertEqualWithAccuracy(expected_values[i], output_values[i], 1e-6);
  }

  ort_api->ReleaseValue(output_tensor);
  ort_api->ReleaseValue(input_tensor);
  ort_api->ReleaseSession(session);
  ort_api->ReleaseSessionOptions(so);
  ort_api->ReleaseEnv(env);
}

- (void)testCppAPI {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

  // initialize session options if needed
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);

  NSString* path = [[NSBundle mainBundle] pathForResource:@"sigmoid" ofType:@"ort"];
  const char* cPath = [path cStringUsingEncoding:NSUTF8StringEncoding];
  Ort::Session session(env, cPath, session_options);

  XCTAssert(true);
}

@end
