// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
//  ios_package_test_cpp_api.mm
//  ios_package_testTests
//

#import <XCTest/XCTest.h>
#include <math.h>
#include <onnxruntime/onnxruntime_cxx_api.h>

@interface ios_package_test_cpp_api : XCTestCase

@end

@implementation ios_package_test_cpp_api

- (void)setUp {
  // Put setup code here. This method is called before the invocation of each test method in the class.
}

- (void)tearDown {
  // Put teardown code here. This method is called after the invocation of each test method in the class.
}

- (void)testCppAPI {
  // This is an e2e test for ORT C++ API
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "testCppAPI");

  // initialize session options if needed
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);

  NSString* path = [[NSBundle mainBundle] pathForResource:@"sigmoid" ofType:@"ort"];
  const char* cPath = [path cStringUsingEncoding:NSUTF8StringEncoding];
  Ort::Session session(env, cPath, session_options);

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

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values, input_tensor_size,
                                                            input_node_dims, 3);
  XCTAssert(input_tensor.IsTensor());

  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names,
                                    &input_tensor, 1, output_node_names, 1);
  XCTAssertEqual(output_tensors.size(), 1);
  XCTAssert(output_tensors.front().IsTensor());

  // Get pointer to output tensor float values
  float* output_values = output_tensors.front().GetTensorMutableData<float>();
  for (size_t i = 0; i < input_tensor_size; i++) {
    NSLog(@"%1.10f\t%1.10f", expected_output_values[i], output_values[i]);
    XCTAssertEqualWithAccuracy(expected_output_values[i], output_values[i], 1e-6);
  }
}

@end
