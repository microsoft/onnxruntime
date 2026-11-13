// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

@import onnxruntime_objc;
@import XCTest;

@interface ORTModuleImportTest : XCTestCase
@end

@implementation ORTModuleImportTest
- (void)testInferenceModuleImport {
  XCTAssertNotNil(ORTVersion());
}
@end
