// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

#import "ort_checkpoint.h"
#import "ort_training_session.h"
#import "ort_env.h"
#import "ort_session.h"

#import "test/test_utils.h"
#import "test/assertion_utils.h"

NS_ASSUME_NONNULL_BEGIN

@interface ORTCheckpointTest : XCTestCase
@property(readonly, nullable) ORTEnv* ortEnv;
@end

@implementation ORTCheckpointTest

- (void)setUp {
  [super setUp];

  self.continueAfterFailure = NO;

  NSError* err = nil;
  _ortEnv = [[ORTEnv alloc] initWithLoggingLevel:ORTLoggingLevelWarning
                                           error:&err];
  ORTAssertNullableResultSuccessful(_ortEnv, err);
}

+ (NSString*)getCheckpointPath {
  NSBundle* bundle = [NSBundle bundleForClass:[ORTCheckpointTest class]];
  NSString* path = [[bundle resourcePath] stringByAppendingPathComponent:@"checkpoint.ckpt"];
  return path;
}

+ (NSString*)getTrainingModelPath {
  NSBundle* bundle = [NSBundle bundleForClass:[ORTCheckpointTest class]];
  NSString* path = [[bundle resourcePath] stringByAppendingPathComponent:@"training_model.onnx"];
  return path;
}

- (void)testSaveCheckpoint {
  NSError* error = nil;
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTCheckpointTest getCheckpointPath] error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);

  // save checkpoint
  NSString* path = [test_utils::createTemporaryDirectory(self) stringByAppendingPathComponent:@"save_checkpoint.ckpt"];
  XCTAssertNotNil(path);
  BOOL result = [checkpoint saveCheckpointToPath:path withOptimizerState:NO error:&error];

  ORTAssertBoolResultSuccessful(result, error);
}

- (void)testInitCheckpoint {
  NSError* error = nil;
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTCheckpointTest getCheckpointPath] error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
}

- (void)testIntProperty {
  NSError* error = nil;
  // Load checkpoint
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTCheckpointTest getCheckpointPath] error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);

  // Add property
  BOOL result = [checkpoint addIntPropertyWithName:@"test" value:314 error:&error];
  ORTAssertBoolResultSuccessful(result, error);

  // Get property
  int64_t value = [checkpoint getIntPropertyWithName:@"test" error:&error];
  XCTAssertEqual(value, 314);
}

- (void)testFloatProperty {
  NSError* error = nil;
  // Load checkpoint
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTCheckpointTest getCheckpointPath] error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);

  // Add property
  BOOL result = [checkpoint addFloatPropertyWithName:@"test" value:3.14f error:&error];
  ORTAssertBoolResultSuccessful(result, error);

  // Get property
  float value = [checkpoint getFloatPropertyWithName:@"test" error:&error];
  XCTAssertEqual(value, 3.14f);
}

- (void)testStringProperty {
  NSError* error = nil;
  // Load checkpoint
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTCheckpointTest getCheckpointPath] error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);

  // Add property
  BOOL result = [checkpoint addStringPropertyWithName:@"test" value:@"hello" error:&error];
  ORTAssertBoolResultSuccessful(result, error);

  // Get property
  NSString* value = [checkpoint getStringPropertyWithName:@"test" error:&error];
  XCTAssertEqualObjects(value, @"hello");
}

- (void)tearDown {
  _ortEnv = nil;

  [super tearDown];
}

@end

NS_ASSUME_NONNULL_END
