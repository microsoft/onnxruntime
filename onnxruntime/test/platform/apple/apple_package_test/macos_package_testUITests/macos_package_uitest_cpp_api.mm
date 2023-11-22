// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
//  macos_package_test_cpp_api.mm
//  macos_package_test_cpp_api
//
//  This file hosts the tests of ORT C++ API
//

#import <XCTest/XCTest.h>
#include <math.h>
#include <onnxruntime/onnxruntime_cxx_api.h>

#if __has_include(<onnxruntime/coreml_provider_factory.h>)
#define COREML_EP_AVAILABLE 1
#else
#define COREML_EP_AVAILABLE 0
#endif

#if COREML_EP_AVAILABLE
#include <onnxruntime/coreml_provider_factory.h>
#endif

void testSigmoid(const char* modelPath, bool useCoreML) {
  // This is an e2e test for ORT C++ API
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "testCppAPI");

  // initialize session options if needed
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);

#if COREML_EP_AVAILABLE
  if (useCoreML) {
    const uint32_t flags = COREML_FLAG_USE_CPU_ONLY;
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, flags));
  }
#else
  (void)useCoreML;
#endif

  Ort::Session session(env, modelPath, session_options);

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
  Ort::Value input_tensor =
      Ort::Value::CreateTensor<float>(memory_info, input_tensor_values, input_tensor_size, input_node_dims, 3);
  XCTAssert(input_tensor.IsTensor());

  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names,
                                    &input_tensor, 1, output_node_names, 1);
  XCTAssertEqual(output_tensors.size(), 1);
  XCTAssert(output_tensors.front().IsTensor());

  // Get pointer to output tensor float values
  float* output_values = output_tensors.front().GetTensorMutableData<float>();
  for (size_t i = 0; i < input_tensor_size; i++) {
    XCTAssertEqualWithAccuracy(expected_output_values[i], output_values[i], 1e-6);
  }
}

@interface macos_package_testUITests : XCTestCase

@end

@implementation macos_package_testUITests

- (void)setUp {
    // Put setup code here. This method is called before the invocation of each test method in the class.

    // In UI tests it is usually best to stop immediately when a failure occurs.
    self.continueAfterFailure = NO;

    // In UI tests it’s important to set the initial state - such as interface orientation - required for your tests before they run. The setUp method is a good place to do this.
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
}

- (NSString*)getFilePath {
  NSBundle* bundle = [NSBundle bundleForClass:[self class]];
  NSString* ns_model_path = [bundle pathForResource:@"sigmoid" ofType:@"ort"];
  XCTAssertNotNil(ns_model_path);
  return ns_model_path;
}

- (void)testCppAPI_Basic {
  testSigmoid([self getFilePath].UTF8String, false /* useCoreML */);
}

#if COREML_EP_AVAILABLE
- (void)testCppAPI_Basic_CoreML {
  testSigmoid([self getFilePath].UTF8String, true /* useCoreML */);
}
#endif

@end
