// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

#import "ort_env.h"
#import "ort_session.h"
#import "ort_value.h"

#include <vector>

NS_ASSUME_NONNULL_BEGIN

@interface ORTSessionTest : XCTestCase

@property(readonly, nullable) ORTEnv* ortEnv;

@end

@implementation ORTSessionTest

- (void)setUp {
  [super setUp];

  self.continueAfterFailure = NO;

  _ortEnv = [[ORTEnv alloc] initWithLoggingLevel:ORTLoggingLevelWarning
                                           error:nil];
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
  ORTValue* ortValue = [[ORTValue alloc] initTensorWithData:data
                                                elementType:ORTTensorElementDataTypeFloat
                                                      shape:shape
                                                      error:&err];
  XCTAssertNotNil(ortValue);
  XCTAssertNil(err);
  return ortValue;
}

- (void)testInitAndRunWithPreallocatedOutputOk {
  NSMutableData* aData = [ORTSessionTest dataWithScalarFloat:1.0f];
  NSMutableData* bData = [ORTSessionTest dataWithScalarFloat:2.0f];
  NSMutableData* cData = [ORTSessionTest dataWithScalarFloat:0.0f];

  ORTValue* a = [ORTSessionTest ortValueWithScalarFloatData:aData];
  ORTValue* b = [ORTSessionTest ortValueWithScalarFloatData:bData];
  ORTValue* c = [ORTSessionTest ortValueWithScalarFloatData:cData];

  NSError* err = nil;
  ORTSession* session = [[ORTSession alloc] initWithEnv:self.ortEnv
                                              modelPath:[ORTSessionTest getAddModelPath]
                                                  error:&err];
  XCTAssertNotNil(session);
  XCTAssertNil(err);

  BOOL runResult = [session runWithInputs:@{@"A" : a, @"B" : b}
                                  outputs:@{@"C" : c}
                                    error:&err];
  XCTAssertTrue(runResult);
  XCTAssertNil(err);

  const float cExpected = 3.0f;
  float cActual;
  memcpy(&cActual, cData.bytes, sizeof(float));
  XCTAssertEqual(cActual, cExpected);
}

- (void)testInitAndRunOk {
  NSMutableData* aData = [ORTSessionTest dataWithScalarFloat:1.0f];
  NSMutableData* bData = [ORTSessionTest dataWithScalarFloat:2.0f];

  ORTValue* a = [ORTSessionTest ortValueWithScalarFloatData:aData];
  ORTValue* b = [ORTSessionTest ortValueWithScalarFloatData:bData];

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

  ORTValue* cOutput = outputs[@"C"];
  XCTAssertNotNil(cOutput);

  NSData* cData = [cOutput tensorDataWithError:&err];
  XCTAssertNotNil(cData);
  XCTAssertNil(err);

  const float cExpected = 3.0f;
  float cActual;
  memcpy(&cActual, cData.bytes, sizeof(float));
  XCTAssertEqual(cActual, cExpected);
}

- (void)testInitFailsWithInvalidPath {
  NSString* invalidModelPath = [ORTSessionTest getTestDataWithRelativePath:@"/invalid/path/to/model.onnx"];
  NSError* err = nil;
  ORTSession* session = [[ORTSession alloc] initWithEnv:self.ortEnv
                                              modelPath:invalidModelPath
                                                  error:&err];
  XCTAssertNil(session);
  XCTAssertNotNil(err);
}

- (void)testRunFailsWithInvalidInput {
  NSMutableData* dData = [ORTSessionTest dataWithScalarFloat:1.0f];
  NSMutableData* cData = [ORTSessionTest dataWithScalarFloat:0.0f];

  ORTValue* d = [ORTSessionTest ortValueWithScalarFloatData:dData];
  ORTValue* c = [ORTSessionTest ortValueWithScalarFloatData:cData];

  NSError* err = nil;
  ORTSession* session = [[ORTSession alloc] initWithEnv:self.ortEnv
                                              modelPath:[ORTSessionTest getAddModelPath]
                                                  error:&err];
  XCTAssertNotNil(session);
  XCTAssertNil(err);

  BOOL runResult = [session runWithInputs:@{@"D" : d}
                                  outputs:@{@"C" : c}
                                    error:&err];
  XCTAssertFalse(runResult);
  XCTAssertNotNil(err);
}

@end

NS_ASSUME_NONNULL_END
