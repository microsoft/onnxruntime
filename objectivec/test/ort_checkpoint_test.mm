// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_APIS
#import <XCTest/XCTest.h>

#import "ort_checkpoint.h"
#import "ort_env.h"
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

- (NSString*)getCheckpointPath {
  NSBundle* bundle = [NSBundle bundleForClass:[ORTCheckpointTest class]];
  NSString* path = [[bundle resourcePath] stringByAppendingPathComponent:@"checkpoint.ckpt"];
  return path;
}

- (void)testLoadCheckpoint {
  NSError* error = nil;
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[self getCheckpointPath] error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
}

- (void)testIntProperty {
  NSError* error = nil;
  // Load checkpoint
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[self getCheckpointPath] error:&error];
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
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[self getCheckpointPath] error:&error];
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
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[self getCheckpointPath] error:&error];
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

#endif  // ENABLE_TRAINING_APIS
