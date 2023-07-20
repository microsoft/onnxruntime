// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

#import "ort_coreml_execution_provider.h"
#import "ort_xnnpack_execution_provider.h"
#import "ort_env.h"
#import "ort_session.h"
#import "ort_value.h"

#import "test/assertion_utils.h"

#include <vector>

NS_ASSUME_NONNULL_BEGIN

@interface ORTSessionTest : XCTestCase

@property(readonly, nullable) ORTEnv* ortEnv;

@end

@implementation ORTSessionTest

- (void)setUp {
  [super setUp];

  self.continueAfterFailure = NO;

  NSError* err = nil;
  _ortEnv = [[ORTEnv alloc] initWithLoggingLevel:ORTLoggingLevelWarning
                                           error:&err];
  ORTAssertNullableResultSuccessful(_ortEnv, err);
}

- (void)tearDown {
  _ortEnv = nil;

  [super tearDown];
}

// model with an Add op
// inputs: A, B
// output: C = A + B
+ (NSString*)getAddModelPath {
  NSBundle* bundle = [NSBundle bundleForClass:[ORTSessionTest class]];
  NSString* path = [bundle pathForResource:@"single_add.basic"
                                    ofType:@"ort"];
  return path;
}

+ (NSString*)getStringModelPath {
  NSBundle* bundle = [NSBundle bundleForClass:[ORTSessionTest class]];
  NSString* path = [bundle pathForResource:@"identity_string"
                                    ofType:@"ort"];
  return path;
}

+ (NSMutableData*)dataWithScalarFloat:(float)value {
  NSMutableData* data = [[NSMutableData alloc] initWithBytes:&value length:sizeof(value)];
  return data;
}

+ (ORTValue*)ortValueWithScalarFloatData:(NSMutableData*)data {
  NSArray<NSNumber*>* shape = @[ @1 ];
  NSError* err = nil;
  ORTValue* ortValue = [[ORTValue alloc] initWithTensorData:data
                                                elementType:ORTTensorElementDataTypeFloat
                                                      shape:shape
                                                      error:&err];
  ORTAssertNullableResultSuccessful(ortValue, err);
  return ortValue;
}

+ (ORTSessionOptions*)makeSessionOptions {
  NSError* err = nil;
  ORTSessionOptions* sessionOptions = [[ORTSessionOptions alloc] initWithError:&err];
  ORTAssertNullableResultSuccessful(sessionOptions, err);
  return sessionOptions;
}

+ (ORTRunOptions*)makeRunOptions {
  NSError* err = nil;
  ORTRunOptions* runOptions = [[ORTRunOptions alloc] initWithError:&err];
  ORTAssertNullableResultSuccessful(runOptions, err);
  return runOptions;
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
                                         sessionOptions:[ORTSessionTest makeSessionOptions]
                                                  error:&err];
  ORTAssertNullableResultSuccessful(session, err);

  BOOL runResult = [session runWithInputs:@{@"A" : a, @"B" : b}
                                  outputs:@{@"C" : c}
                               runOptions:[ORTSessionTest makeRunOptions]
                                    error:&err];
  ORTAssertBoolResultSuccessful(runResult, err);

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
                                         sessionOptions:[ORTSessionTest makeSessionOptions]
                                                  error:&err];
  ORTAssertNullableResultSuccessful(session, err);

  NSDictionary<NSString*, ORTValue*>* outputs =
      [session runWithInputs:@{@"A" : a, @"B" : b}
                 outputNames:[NSSet setWithArray:@[ @"C" ]]
                  runOptions:[ORTSessionTest makeRunOptions]
                       error:&err];
  ORTAssertNullableResultSuccessful(outputs, err);

  ORTValue* cOutput = outputs[@"C"];
  XCTAssertNotNil(cOutput);

  NSData* cData = [cOutput tensorDataWithError:&err];
  ORTAssertNullableResultSuccessful(cData, err);

  const float cExpected = 3.0f;
  float cActual;
  memcpy(&cActual, cData.bytes, sizeof(float));
  XCTAssertEqual(cActual, cExpected);
}

- (void)testGetNamesOk {
  NSError* err = nil;
  ORTSession* session = [[ORTSession alloc] initWithEnv:self.ortEnv
                                              modelPath:[ORTSessionTest getAddModelPath]
                                         sessionOptions:[ORTSessionTest makeSessionOptions]
                                                  error:&err];
  ORTAssertNullableResultSuccessful(session, err);

  NSArray<NSString*>* inputNames = [session inputNamesWithError:&err];
  ORTAssertNullableResultSuccessful(inputNames, err);
  XCTAssertEqualObjects(inputNames, (@[ @"A", @"B" ]));

  NSArray<NSString*>* overridableInitializerNames = [session overridableInitializerNamesWithError:&err];
  ORTAssertNullableResultSuccessful(overridableInitializerNames, err);
  XCTAssertEqualObjects(overridableInitializerNames, (@[]));

  NSArray<NSString*>* outputNames = [session outputNamesWithError:&err];
  ORTAssertNullableResultSuccessful(outputNames, err);
  XCTAssertEqualObjects(outputNames, (@[ @"C" ]));
}

- (void)testInitFailsWithInvalidPath {
  NSString* invalidModelPath = @"invalid/path/to/model.ort";
  NSError* err = nil;
  ORTSession* session = [[ORTSession alloc] initWithEnv:self.ortEnv
                                              modelPath:invalidModelPath
                                         sessionOptions:[ORTSessionTest makeSessionOptions]
                                                  error:&err];
  ORTAssertNullableResultUnsuccessful(session, err);
}

- (void)testRunFailsWithInvalidInput {
  NSMutableData* dData = [ORTSessionTest dataWithScalarFloat:1.0f];
  NSMutableData* cData = [ORTSessionTest dataWithScalarFloat:0.0f];

  ORTValue* d = [ORTSessionTest ortValueWithScalarFloatData:dData];
  ORTValue* c = [ORTSessionTest ortValueWithScalarFloatData:cData];

  NSError* err = nil;
  ORTSession* session = [[ORTSession alloc] initWithEnv:self.ortEnv
                                              modelPath:[ORTSessionTest getAddModelPath]
                                         sessionOptions:[ORTSessionTest makeSessionOptions]
                                                  error:&err];
  ORTAssertNullableResultSuccessful(session, err);

  BOOL runResult = [session runWithInputs:@{@"D" : d}
                                  outputs:@{@"C" : c}
                               runOptions:[ORTSessionTest makeRunOptions]
                                    error:&err];
  ORTAssertBoolResultUnsuccessful(runResult, err);
}

- (void)testAppendCoreMLEP {
  NSError* err = nil;
  ORTSessionOptions* sessionOptions = [ORTSessionTest makeSessionOptions];
  ORTCoreMLExecutionProviderOptions* coreMLOptions = [[ORTCoreMLExecutionProviderOptions alloc] init];
  coreMLOptions.enableOnSubgraphs = YES;  // set an arbitrary option

  BOOL appendResult = [sessionOptions appendCoreMLExecutionProviderWithOptions:coreMLOptions
                                                                         error:&err];

  if (!ORTIsCoreMLExecutionProviderAvailable()) {
    ORTAssertBoolResultUnsuccessful(appendResult, err);
    return;
  }

  ORTAssertBoolResultSuccessful(appendResult, err);

  ORTSession* session = [[ORTSession alloc] initWithEnv:self.ortEnv
                                              modelPath:[ORTSessionTest getAddModelPath]
                                         sessionOptions:sessionOptions
                                                  error:&err];
  ORTAssertNullableResultSuccessful(session, err);
}

- (void)testAppendXnnpackEP {
  NSError* err = nil;
  ORTSessionOptions* sessionOptions = [ORTSessionTest makeSessionOptions];
  ORTXnnpackExecutionProviderOptions* XnnpackOptions = [[ORTXnnpackExecutionProviderOptions alloc] init];
  XnnpackOptions.intra_op_num_threads = 2;

  BOOL appendResult = [sessionOptions appendXnnpackExecutionProviderWithOptions:XnnpackOptions
                                                                          error:&err];
  // Without xnnpack EP in building also can pass the test
  NSString* err_msg = [err localizedDescription];
  if (!appendResult && [err_msg containsString:@"XNNPACK execution provider is not supported in this build. "]) {
    return;
  }

  ORTAssertBoolResultSuccessful(appendResult, err);

  ORTSession* session = [[ORTSession alloc] initWithEnv:self.ortEnv
                                              modelPath:[ORTSessionTest getAddModelPath]
                                         sessionOptions:sessionOptions
                                                  error:&err];
  ORTAssertNullableResultSuccessful(session, err);
}

static bool gDummyRegisterCustomOpsFnCalled = false;

static OrtStatus* _Nullable DummyRegisterCustomOpsFn(OrtSessionOptions* /*session_options*/,
                                                     const OrtApiBase* /*api*/) {
  gDummyRegisterCustomOpsFnCalled = true;
  return nullptr;
}

- (void)testRegisterCustomOpsUsingFunctionPointer {
  NSError* err = nil;
  ORTSessionOptions* sessionOptions = [ORTSessionTest makeSessionOptions];

  gDummyRegisterCustomOpsFnCalled = false;
  BOOL registerResult = [sessionOptions registerCustomOpsUsingFunctionPointer:&DummyRegisterCustomOpsFn
                                                                        error:&err];
  ORTAssertBoolResultSuccessful(registerResult, err);

  XCTAssertEqual(gDummyRegisterCustomOpsFnCalled, true);
}

- (void)testStringInputs {
  NSError* err = nil;
  NSArray<NSString*>* stringData = @[ @"ONNX Runtime", @"is the", @"best", @"AI Framework" ];
  ORTValue* stringValue = [[ORTValue alloc] initWithTensorStringData:stringData shape:@[ @2, @2 ] error:&err];
  ORTAssertNullableResultSuccessful(stringValue, err);

  ORTSession* session = [[ORTSession alloc] initWithEnv:self.ortEnv
                                              modelPath:[ORTSessionTest getStringModelPath]
                                         sessionOptions:[ORTSessionTest makeSessionOptions]
                                                  error:&err];
  ORTAssertNullableResultSuccessful(session, err);

  NSDictionary<NSString*, ORTValue*>* outputs =
      [session runWithInputs:@{@"input:0" : stringValue}
                 outputNames:[NSSet setWithArray:@[ @"output:0" ]]
                  runOptions:[ORTSessionTest makeRunOptions]
                       error:&err];
  ORTAssertNullableResultSuccessful(outputs, err);

  ORTValue* outputStringValue = outputs[@"output:0"];
  XCTAssertNotNil(outputStringValue);

  NSArray<NSString*>* outputStringData = [outputStringValue tensorStringDataWithError:&err];
  ORTAssertNullableResultSuccessful(outputStringData, err);

  XCTAssertEqual([stringData count], [outputStringData count]);
  XCTAssertTrue([stringData isEqualToArray:outputStringData]);
}

@end

NS_ASSUME_NONNULL_END
