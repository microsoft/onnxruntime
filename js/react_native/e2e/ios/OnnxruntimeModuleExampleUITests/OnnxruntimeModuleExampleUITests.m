// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

@interface OnnxruntimeModuleExampleUITests : XCTestCase

@end

@implementation OnnxruntimeModuleExampleUITests

- (void)setUp {
  self.continueAfterFailure = NO;
}

- (void)testExample {
  XCUIApplication *app = [[XCUIApplication alloc] init];
  [app launch];

  XCTAssert([app.textFields[@"output"] waitForExistenceWithTimeout:180]);
  NSString* value = app.textFields[@"output"].value;
  XCTAssertEqualObjects(value, @"Result: 3");
}

@end
