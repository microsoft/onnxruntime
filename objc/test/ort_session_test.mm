// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

#import "onnxruntime/ort_env.h"
#import "onnxruntime/ort_session.h"
#import "onnxruntime/ort_value.h"

#include <vector>

@interface ORTSessionTest : XCTestCase

@property(readonly) ORTEnv* ortEnv;

@end

static NSString* kTestDataDir = [NSString stringWithFormat:@"%@/Contents/Resources/testdata",
                                                           [[NSBundle bundleForClass:[ORTSessionTest class]] bundlePath]];

// model with an Add op
// inputs: A, B
// output: C = A + B
static NSString* kAddModelPath = [kTestDataDir stringByAppendingString:@"/single_add.onnx"];

@implementation ORTSessionTest

- (BOOL)setUpWithError:(NSError**)error {
  _ortEnv = [[ORTEnv alloc] initWithError:error];
  if (!_ortEnv) {
    return NO;
  }
  return YES;
}

- (BOOL)tearDownWithError:(NSError**)error {
  _ortEnv = nil;
  return YES;
}

+ (NSMutableData*)dataWithScalarFloat:(float)value {
  NSMutableData* data = [[NSMutableData alloc] initWithBytes:&value length:sizeof(value)];
  return data;
}

+ (ORTValue*)ortValueWithScalarFloatData:(NSMutableData*)data {
  const std::vector<int64_t> shape{1};
  NSError* err;
  ORTValue* ort_value = [[ORTValue alloc] initTensorWithData:data
                                                 elementType:ORTTensorElementDataTypeFloat
                                                       shape:shape.data()
                                                    shapeLen:shape.size()
                                                       error:&err];
  XCTAssertNotNil(ort_value);
  XCTAssertNil(err);
  return ort_value;
}

- (void)testInitAndRunOk {
  NSMutableData* a_data = [ORTSessionTest dataWithScalarFloat:1.0f];
  NSMutableData* b_data = [ORTSessionTest dataWithScalarFloat:2.0f];
  NSMutableData* c_data = [ORTSessionTest dataWithScalarFloat:0.0f];

  ORTValue* a = [ORTSessionTest ortValueWithScalarFloatData:a_data];
  ORTValue* b = [ORTSessionTest ortValueWithScalarFloatData:b_data];
  ORTValue* c = [ORTSessionTest ortValueWithScalarFloatData:c_data];

  NSError* err;
  ORTSession* session = [[ORTSession alloc] initWithEnv:self.ortEnv
                                              modelPath:kAddModelPath
                                                  error:&err];
  XCTAssertNotNil(session);
  XCTAssertNil(err);

  BOOL run_result = [session runWithInputs:@{@"A" : a, @"B" : b}
                                   outputs:@{@"C" : c}
                                     error:&err];
  XCTAssertTrue(run_result);
  XCTAssertNil(err);

  const float c_expected = 3.0f;
  float c_actual;
  memcpy(&c_actual, c_data.bytes, sizeof(float));
  XCTAssertEqual(c_actual, c_expected);
}

- (void)testInitFailsWithInvalidPath {
  NSString* invalid_model_path = [kTestDataDir stringByAppendingString:@"/invalid/path/to/model.onnx"];
  NSError* err;
  ORTSession* session = [[ORTSession alloc] initWithEnv:self.ortEnv
                                              modelPath:invalid_model_path
                                                  error:&err];
  XCTAssertNil(session);
  XCTAssertNotNil(err);
}

- (void)testRunFailsWithInvalidInput {
  NSMutableData* d_data = [ORTSessionTest dataWithScalarFloat:1.0f];
  NSMutableData* c_data = [ORTSessionTest dataWithScalarFloat:0.0f];

  ORTValue* d = [ORTSessionTest ortValueWithScalarFloatData:d_data];
  ORTValue* c = [ORTSessionTest ortValueWithScalarFloatData:c_data];

  NSError* err;
  ORTSession* session = [[ORTSession alloc] initWithEnv:self.ortEnv
                                              modelPath:kAddModelPath
                                                  error:&err];
  XCTAssertNotNil(session);
  XCTAssertNil(err);

  BOOL run_result = [session runWithInputs:@{@"D" : d}
                                   outputs:@{@"C" : c}
                                     error:&err];
  XCTAssertFalse(run_result);
  XCTAssertNotNil(err);
}

@end
