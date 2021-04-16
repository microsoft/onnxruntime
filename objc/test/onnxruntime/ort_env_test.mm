// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

#import "onnxruntime/ort_env.h"

@interface ORTEnvTest : XCTestCase
@end

@implementation ORTEnvTest

- (void)testCreateEnv {
    ORTEnv* env = [[ORTEnv alloc] initWithError:nil];
    XCTAssertNotNil(env);
}

@end
