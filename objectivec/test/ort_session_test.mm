// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

#import "ort_coreml_execution_provider.h"
#import "ort_xnnpack_execution_provider.h"
#import "ort_env.h"
#import "ort_session.h"
#import "ort_value.h"

#import "cxx_api.h"
#import "ort_session_internal.h"
#import "test/assertion_utils.h"

#include <limits>
#include <string>
#include <vector>

NS_ASSUME_NONNULL_BEGIN

namespace {
struct EpContextReadRegistration {
  OrtReadNamedBufferFunc function = nullptr;
  void* state = nullptr;
  size_t maxDataSize = 0;
};

EpContextReadRegistration GetEpContextReadRegistration(ORTSessionOptions* sessionOptions) {
  Ort::EpContextConfig config{[sessionOptions CXXAPIOrtSessionOptions]};
  EpContextReadRegistration registration;
  config.GetReadFunc(registration.function, registration.state, registration.maxDataSize);
  return registration;
}

NSString* StatusMessage(const Ort::Status& status) {
  const std::string message = status.GetErrorMessage();
  NSString* result = [NSString stringWithUTF8String:message.c_str()];
  return result ?: @"";
}
}  // namespace

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

- (void)testAppendCoreMLEP_v2 {
  NSError* err = nil;
  ORTSessionOptions* sessionOptions = [ORTSessionTest makeSessionOptions];
  NSDictionary* provider_options = @{@"EnableOnSubgraphs" : @"1"};  // set an arbitrary option

  BOOL appendResult = [sessionOptions appendCoreMLExecutionProviderWithOptionsV2:provider_options
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

- (void)testEpContextDataReadBlockReturnsData {
  ORTSessionOptions* sessionOptions = [ORTSessionTest makeSessionOptions];
  NSData* expectedData = [@"epcontext" dataUsingEncoding:NSUTF8StringEncoding];
  __block NSString* receivedName = nil;
  NSError* err = nil;
  BOOL result = [sessionOptions
      setEpContextDataReadBlock:^NSData*(NSString* name, NSError** /*error*/) {
        receivedName = name;
        return expectedData;
      }
                    maxDataSize:1024
                          error:&err];
  ORTAssertBoolResultSuccessful(result, err);

  const EpContextReadRegistration registration = GetEpContextReadRegistration(sessionOptions);
  XCTAssertNotEqual(registration.function, nullptr);
  XCTAssertNotEqual(registration.state, nullptr);
  XCTAssertEqual(registration.maxDataSize, 1024U);

  Ort::AllocatorWithDefaultOptions allocator;
  void* buffer = reinterpret_cast<void*>(0x1);
  size_t dataSize = 1;
  Ort::Status status{
      registration.function(registration.state, "model.ctx", allocator, &buffer, &dataSize)};
  XCTAssertTrue(status.IsOK(), @"%@", StatusMessage(status));
  XCTAssertEqualObjects(receivedName, @"model.ctx");
  XCTAssertEqual(dataSize, expectedData.length);

  NSData* actualData = [NSData dataWithBytes:buffer length:dataSize];
  allocator.Free(buffer);
  XCTAssertEqualObjects(actualData, expectedData);
}

- (void)testEpContextDataReadBlockAllowsEmptyData {
  ORTSessionOptions* sessionOptions = [ORTSessionTest makeSessionOptions];
  NSError* err = nil;
  BOOL result = [sessionOptions
      setEpContextDataReadBlock:^NSData*(NSString* /*name*/, NSError** /*error*/) {
        return [NSData data];
      }
                    maxDataSize:1
                          error:&err];
  ORTAssertBoolResultSuccessful(result, err);

  const EpContextReadRegistration registration = GetEpContextReadRegistration(sessionOptions);
  Ort::AllocatorWithDefaultOptions allocator;
  void* buffer = reinterpret_cast<void*>(0x1);
  size_t dataSize = 1;
  Ort::Status status{
      registration.function(registration.state, "empty.ctx", allocator, &buffer, &dataSize)};
  XCTAssertTrue(status.IsOK(), @"%@", StatusMessage(status));
  XCTAssertEqual(buffer, nullptr);
  XCTAssertEqual(dataSize, 0U);
}

- (void)testEpContextDataReadBlockRejectsInvalidAndOversizedLimits {
  ORTSessionOptions* sessionOptions = [ORTSessionTest makeSessionOptions];
  ORTEpContextDataReadBlock block = ^NSData*(NSString* /*name*/, NSError** /*error*/) {
    return [@"large" dataUsingEncoding:NSUTF8StringEncoding];
  };

  NSError* err = nil;
  XCTAssertFalse([sessionOptions setEpContextDataReadBlock:block maxDataSize:0 error:&err]);
  XCTAssertNotNil(err);

  err = nil;
  XCTAssertFalse([sessionOptions setEpContextDataReadBlock:block maxDataSize:NSUIntegerMax error:&err]);
  XCTAssertNotNil(err);

  err = nil;
  BOOL result = [sessionOptions setEpContextDataReadBlock:block maxDataSize:4 error:&err];
  ORTAssertBoolResultSuccessful(result, err);

  const EpContextReadRegistration registration = GetEpContextReadRegistration(sessionOptions);
  Ort::AllocatorWithDefaultOptions allocator;
  void* buffer = reinterpret_cast<void*>(0x1);
  size_t dataSize = 1;
  Ort::Status status{
      registration.function(registration.state, "large.ctx", allocator, &buffer, &dataSize)};
  XCTAssertFalse(status.IsOK());
  XCTAssertEqual(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
  XCTAssertTrue([StatusMessage(status) containsString:@"exceeds maxDataSize"]);
  XCTAssertEqual(buffer, nullptr);
  XCTAssertEqual(dataSize, 0U);
}

- (void)testEpContextDataReadBlockPropagatesNSError {
  ORTSessionOptions* sessionOptions = [ORTSessionTest makeSessionOptions];
  NSError* err = nil;
  BOOL result = [sessionOptions
      setEpContextDataReadBlock:^NSData*(NSString* /*name*/, NSError** error) {
        *error = [NSError errorWithDomain:@"EpContextReadTest"
                                     code:7
                                 userInfo:@{NSLocalizedDescriptionKey : @"read failed"}];
        return nil;
      }
                    maxDataSize:1024
                          error:&err];
  ORTAssertBoolResultSuccessful(result, err);

  const EpContextReadRegistration registration = GetEpContextReadRegistration(sessionOptions);
  Ort::AllocatorWithDefaultOptions allocator;
  void* buffer = reinterpret_cast<void*>(0x1);
  size_t dataSize = 1;
  Ort::Status status{
      registration.function(registration.state, "error.ctx", allocator, &buffer, &dataSize)};
  XCTAssertFalse(status.IsOK());
  XCTAssertEqual(status.GetErrorCode(), ORT_FAIL);
  XCTAssertTrue([StatusMessage(status) containsString:@"read failed"]);
  XCTAssertEqual(buffer, nullptr);
  XCTAssertEqual(dataSize, 0U);
}

- (void)testEpContextDataReadBlockRejectsNilWithoutNSError {
  ORTSessionOptions* sessionOptions = [ORTSessionTest makeSessionOptions];
  NSError* err = nil;
  BOOL result = [sessionOptions
      setEpContextDataReadBlock:^NSData*(NSString* /*name*/, NSError** /*error*/) {
        return nil;
      }
                    maxDataSize:1024
                          error:&err];
  ORTAssertBoolResultSuccessful(result, err);

  const EpContextReadRegistration registration = GetEpContextReadRegistration(sessionOptions);
  Ort::AllocatorWithDefaultOptions allocator;
  void* buffer = reinterpret_cast<void*>(0x1);
  size_t dataSize = 1;
  Ort::Status status{
      registration.function(registration.state, "nil.ctx", allocator, &buffer, &dataSize)};
  XCTAssertFalse(status.IsOK());
  XCTAssertEqual(status.GetErrorCode(), ORT_FAIL);
  XCTAssertTrue([StatusMessage(status) containsString:@"without setting an error"]);
  XCTAssertEqual(buffer, nullptr);
  XCTAssertEqual(dataSize, 0U);
}

- (void)testEpContextDataReadBlockRejectsInvalidData {
  ORTSessionOptions* sessionOptions = [ORTSessionTest makeSessionOptions];
  NSError* err = nil;
  BOOL result = [sessionOptions
      setEpContextDataReadBlock:^NSData*(NSString* /*name*/, NSError** /*error*/) {
        return (NSData*)(id) @"not data";
      }
                    maxDataSize:1024
                          error:&err];
  ORTAssertBoolResultSuccessful(result, err);

  const EpContextReadRegistration registration = GetEpContextReadRegistration(sessionOptions);
  Ort::AllocatorWithDefaultOptions allocator;
  void* buffer = reinterpret_cast<void*>(0x1);
  size_t dataSize = 1;
  Ort::Status status{
      registration.function(registration.state, "invalid.ctx", allocator, &buffer, &dataSize)};
  XCTAssertFalse(status.IsOK());
  XCTAssertEqual(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
  XCTAssertTrue([StatusMessage(status) containsString:@"not NSData"]);
  XCTAssertEqual(buffer, nullptr);
  XCTAssertEqual(dataSize, 0U);
}

- (void)testEpContextDataReadBlockContainsNSException {
  ORTSessionOptions* sessionOptions = [ORTSessionTest makeSessionOptions];
  NSError* err = nil;
  BOOL result = [sessionOptions
      setEpContextDataReadBlock:^NSData*(NSString* /*name*/, NSError** /*error*/) {
        @throw [NSException exceptionWithName:@"EpContextReadException"
                                       reason:@"callback failed"
                                     userInfo:nil];
      }
                    maxDataSize:1024
                          error:&err];
  ORTAssertBoolResultSuccessful(result, err);

  const EpContextReadRegistration registration = GetEpContextReadRegistration(sessionOptions);
  Ort::AllocatorWithDefaultOptions allocator;
  void* buffer = reinterpret_cast<void*>(0x1);
  size_t dataSize = 1;
  Ort::Status status{
      registration.function(registration.state, "exception.ctx", allocator, &buffer, &dataSize)};
  XCTAssertFalse(status.IsOK());
  XCTAssertEqual(status.GetErrorCode(), ORT_FAIL);
  XCTAssertTrue([StatusMessage(status) containsString:@"Objective-C exception"]);
  XCTAssertEqual(buffer, nullptr);
  XCTAssertEqual(dataSize, 0U);
}

- (void)testEpContextDataReadBlockCanBeReplacedAndCleared {
  ORTSessionOptions* sessionOptions = [ORTSessionTest makeSessionOptions];
  NSError* err = nil;
  BOOL result = [sessionOptions
      setEpContextDataReadBlock:^NSData*(NSString* /*name*/, NSError** /*error*/) {
        return [@"first" dataUsingEncoding:NSUTF8StringEncoding];
      }
                    maxDataSize:16
                          error:&err];
  ORTAssertBoolResultSuccessful(result, err);

  err = nil;
  result = [sessionOptions
      setEpContextDataReadBlock:^NSData*(NSString* /*name*/, NSError** /*error*/) {
        return [@"discarded" dataUsingEncoding:NSUTF8StringEncoding];
      }
                    maxDataSize:0
                          error:&err];
  XCTAssertFalse(result);
  XCTAssertNotNil(err);

  EpContextReadRegistration registration = GetEpContextReadRegistration(sessionOptions);
  XCTAssertEqual(registration.maxDataSize, 16U);

  Ort::AllocatorWithDefaultOptions allocator;
  void* buffer = nullptr;
  size_t dataSize = 0;
  Ort::Status status{
      registration.function(registration.state, "original.ctx", allocator, &buffer, &dataSize)};
  XCTAssertTrue(status.IsOK(), @"%@", StatusMessage(status));
  NSData* actualData = [NSData dataWithBytes:buffer length:dataSize];
  allocator.Free(buffer);
  XCTAssertEqualObjects(actualData, [@"first" dataUsingEncoding:NSUTF8StringEncoding]);

  err = nil;
  result = [sessionOptions
      setEpContextDataReadBlock:^NSData*(NSString* /*name*/, NSError** /*error*/) {
        return [@"second" dataUsingEncoding:NSUTF8StringEncoding];
      }
                    maxDataSize:8
                          error:&err];
  ORTAssertBoolResultSuccessful(result, err);

  registration = GetEpContextReadRegistration(sessionOptions);
  XCTAssertEqual(registration.maxDataSize, 8U);

  buffer = nullptr;
  dataSize = 0;
  status = Ort::Status{
      registration.function(registration.state, "replacement.ctx", allocator, &buffer, &dataSize)};
  XCTAssertTrue(status.IsOK(), @"%@", StatusMessage(status));
  actualData = [NSData dataWithBytes:buffer length:dataSize];
  allocator.Free(buffer);
  XCTAssertEqualObjects(actualData, [@"second" dataUsingEncoding:NSUTF8StringEncoding]);

  result = [sessionOptions clearEpContextDataReadBlockWithError:&err];
  ORTAssertBoolResultSuccessful(result, err);

  registration = GetEpContextReadRegistration(sessionOptions);
  XCTAssertEqual(registration.function, nullptr);
  XCTAssertEqual(registration.state, nullptr);
  XCTAssertEqual(registration.maxDataSize, std::numeric_limits<size_t>::max());
}

- (void)testSessionRetainsEpContextDataReadBlockRegistration {
  ORTSessionOptions* sessionOptions = [ORTSessionTest makeSessionOptions];
  NSObject* __weak weakCapturedObject = nil;
  NSError* err = nil;
  BOOL result;
  @autoreleasepool {
    NSObject* capturedObject = [[NSObject alloc] init];
    weakCapturedObject = capturedObject;
    result = [sessionOptions
        setEpContextDataReadBlock:^NSData*(NSString* /*name*/, NSError** /*error*/) {
          (void)capturedObject;
          return [NSData data];
        }
                      maxDataSize:1
                            error:&err];
    ORTAssertBoolResultSuccessful(result, err);
  }
  XCTAssertNotNil(weakCapturedObject);

  ORTSession* session = [[ORTSession alloc] initWithEnv:self.ortEnv
                                              modelPath:[ORTSessionTest getAddModelPath]
                                         sessionOptions:sessionOptions
                                                  error:&err];
  ORTAssertNullableResultSuccessful(session, err);

  result = [sessionOptions clearEpContextDataReadBlockWithError:&err];
  ORTAssertBoolResultSuccessful(result, err);
  sessionOptions = nil;
  XCTAssertNotNil(weakCapturedObject);

  session = nil;
  XCTAssertNil(weakCapturedObject);
}

- (void)testFailedSessionConstructionReleasesEpContextDataReadBlockSnapshot {
  ORTSessionOptions* sessionOptions = [ORTSessionTest makeSessionOptions];
  NSObject* __weak weakCapturedObject = nil;
  NSError* err = nil;
  @autoreleasepool {
    NSObject* capturedObject = [[NSObject alloc] init];
    weakCapturedObject = capturedObject;
    BOOL result = [sessionOptions
        setEpContextDataReadBlock:^NSData*(NSString* /*name*/, NSError** /*error*/) {
          (void)capturedObject;
          return [NSData data];
        }
                      maxDataSize:1
                            error:&err];
    ORTAssertBoolResultSuccessful(result, err);
  }
  XCTAssertNotNil(weakCapturedObject);

  ORTSession* session = [[ORTSession alloc] initWithEnv:self.ortEnv
                                              modelPath:@"invalid/path/to/model.onnx"
                                         sessionOptions:sessionOptions
                                                  error:&err];
  ORTAssertNullableResultUnsuccessful(session, err);

  sessionOptions = nil;
  XCTAssertNil(weakCapturedObject);
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

- (void)testKeepORTEnvReference {
  ORTEnv* __weak envWeak = _ortEnv;
  // Remove sole strong reference to the ORTEnv created in setUp.
  _ortEnv = nil;
  // There should be no more strong references to it.
  XCTAssertNil(envWeak);

  // Create a new ORTEnv.
  NSError* err = nil;
  ORTEnv* env = [[ORTEnv alloc] initWithLoggingLevel:ORTLoggingLevelWarning
                                               error:&err];
  ORTAssertNullableResultSuccessful(env, err);

  ORTSession* session = [[ORTSession alloc] initWithEnv:env
                                              modelPath:[ORTSessionTest getAddModelPath]
                                         sessionOptions:[ORTSessionTest makeSessionOptions]
                                                  error:&err];
  ORTAssertNullableResultSuccessful(session, err);

  envWeak = env;
  // Remove strong reference to the ORTEnv passed to the ORTSession initializer.
  env = nil;
  // ORTSession should keep a strong reference to it.
  XCTAssertNotNil(envWeak);
}

@end

NS_ASSUME_NONNULL_END
