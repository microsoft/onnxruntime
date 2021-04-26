// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

#import "onnxruntime/ort_env.h"

NS_ASSUME_NONNULL_BEGIN

@interface ORTEnvTest : XCTestCase
@end

@implementation ORTEnvTest

- (void)testInitOk {
  NSError* err = nil;
  ORTEnv* env = [[ORTEnv alloc] initWithError:&err];
  XCTAssertNotNil(env);
  XCTAssertNil(err);
}

@end

NS_ASSUME_NONNULL_END
