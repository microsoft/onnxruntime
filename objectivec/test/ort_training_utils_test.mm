// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>
#import "ort_training_session.h"

NS_ASSUME_NONNULL_BEGIN

@interface ORTTrainingUtilsTest : XCTestCase
@end

@implementation ORTTrainingUtilsTest

- (void)setUp {
  [super setUp];

  self.continueAfterFailure = NO;
}

- (void)testSetSeed {
  ORTSetSeed(2718);
}

@end

NS_ASSUME_NONNULL_END
