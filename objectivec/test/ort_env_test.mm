// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

#import "ort_env.h"

#import "test/assertion_utils.h"

NS_ASSUME_NONNULL_BEGIN

@interface ORTEnvTest : XCTestCase
@end

@implementation ORTEnvTest

- (void)testInitOk {
  NSError* err = nil;
  ORTEnv* env = [[ORTEnv alloc] initWithLoggingLevel:ORTLoggingLevelWarning
                                               error:&err];
  ORTAssertNullableResultSuccessful(env, err);
}

@end

NS_ASSUME_NONNULL_END
