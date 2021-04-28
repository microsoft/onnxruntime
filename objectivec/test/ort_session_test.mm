// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

#import "onnxruntime/ort_env.h"
#import "onnxruntime/ort_session.h"
#import "onnxruntime/ort_value.h"

#include <vector>

NS_ASSUME_NONNULL_BEGIN

@interface ORTSessionTest : XCTestCase

@property(readonly, nullable) ORTEnv* ortEnv;

@end

@implementation ORTSessionTest

- (void)setUp {
  [super setUp];

  self.continueAfterFailure = NO;

  _ortEnv = [[ORTEnv alloc] initWithError:nil];
  XCTAssertNotNil(_ortEnv);
}

- (void)tearDown {
  _ortEnv = nil;

  [super tearDown];
}

+ (NSString*)getTestDataWithRelativePath:(NSString*)relativePath {
  NSString* testDataDir = [NSString stringWithFormat:@"%@/Contents/Resources/testdata",
                                                     [[NSBundle bundleForClass:[ORTSessionTest class]] bundlePath]];
  return [testDataDir stringByAppendingString:relativePath];
}

// model with an Add op
// inputs: A, B
// output: C = A + B
+ (NSString*)getAddModelPath {
  return [ORTSessionTest getTestDataWithRelativePath:@"/single_add.onnx"];
}

+ (NSMutableData*)dataWithScalarFloat:(float)value {
  NSMutableData* data = [[NSMutableData alloc] initWithBytes:&value length:sizeof(value)];
  return data;
}

+ (ORTValue*)ortValueWithScalarFloatData:(NSMutableData*)data {
  NSArray<NSNumber*>* shape = @[ @1 ];
  NSError* err = nil;
  ORTValue* ort_value = [[ORTValue alloc] initTensorWithData:data
                                                 elementType:ORTTensorElementDataTypeFloat
                                                       shape:shape
                                                       error:&err];
  XCTAssertNotNil(ort_value);
  XCTAssertNil(err);
  return ort_value;
}

- (void)testInitAndRunWithPreallocatedOutputOk {
  NSMutableData* a_data = [ORTSessionTest dataWithScalarFloat:1.0f];
  NSMutableData* b_data = [ORTSessionTest dataWithScalarFloat:2.0f];
  NSMutableData* c_data = [ORTSessionTest dataWithScalarFloat:0.0f];

  ORTValue* a = [ORTSessionTest ortValueWithScalarFloatData:a_data];
  ORTValue* b = [ORTSessionTest ortValueWithScalarFloatData:b_data];
  ORTValue* c = [ORTSessionTest ortValueWithScalarFloatData:c_data];

  NSError* err = nil;
  ORTSession* session = [[ORTSession alloc] initWithEnv:self.ortEnv
                                              modelPath:[ORTSessionTest getAddModelPath]
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

- (void)testInitAndRunOk {
  NSMutableData* a_data = [ORTSessionTest dataWithScalarFloat:1.0f];
  NSMutableData* b_data = [ORTSessionTest dataWithScalarFloat:2.0f];

  ORTValue* a = [ORTSessionTest ortValueWithScalarFloatData:a_data];
  ORTValue* b = [ORTSessionTest ortValueWithScalarFloatData:b_data];

  NSError* err = nil;
  ORTSession* session = [[ORTSession alloc] initWithEnv:self.ortEnv
                                              modelPath:[ORTSessionTest getAddModelPath]
                                                  error:&err];
  XCTAssertNotNil(session);
  XCTAssertNil(err);

  NSDictionary<NSString*, ORTValue*>* outputs =
      [session runWithInputs:@{@"A" : a, @"B" : b}
                 outputNames:[NSSet setWithArray:@[ @"C" ]]
                       error:&err];
  XCTAssertNotNil(outputs);
  XCTAssertNil(err);

  ORTValue* c_output = outputs[@"C"];
  XCTAssertNotNil(c_output);

  NSData* c_data = [c_output tensorDataWithError:&err];
  XCTAssertNotNil(c_data);
  XCTAssertNil(err);

  const float c_expected = 3.0f;
  float c_actual;
  memcpy(&c_actual, c_data.bytes, sizeof(float));
  XCTAssertEqual(c_actual, c_expected);
}

- (void)testInitFailsWithInvalidPath {
  NSString* invalid_model_path = [ORTSessionTest getTestDataWithRelativePath:@"/invalid/path/to/model.onnx"];
  NSError* err = nil;
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

  NSError* err = nil;
  ORTSession* session = [[ORTSession alloc] initWithEnv:self.ortEnv
                                              modelPath:[ORTSessionTest getAddModelPath]
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

NS_ASSUME_NONNULL_END
