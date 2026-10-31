// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

#include <stdexcept>

#import "error_utils.h"

NS_ASSUME_NONNULL_BEGIN

@interface ORTErrorUtilsTest : XCTestCase
@end

@implementation ORTErrorUtilsTest

- (void)testSaveUTF8Description {
  NSError* error = nil;
  ORTSaveCodeAndDescriptionToError(ORT_INVALID_ARGUMENT, "An error: caf\xC3\xA9", &error);

  XCTAssertNotNil(error);
  XCTAssertEqualObjects(error.domain, @"onnxruntime");
  XCTAssertEqual(error.code, ORT_INVALID_ARGUMENT);
  XCTAssertEqualObjects(error.localizedDescription, @"An error: caf\u00E9");
}

- (void)testSaveEmptyDescription {
  NSError* error = nil;
  ORTSaveCodeAndDescriptionToError(ORT_FAIL, "", &error);

  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, ORT_FAIL);
  XCTAssertEqualObjects(error.localizedDescription, @"");
}

- (void)testSaveInvalidUTF8Description {
  NSError* error = nil;
  ORTSaveCodeAndDescriptionToError(ORT_INVALID_ARGUMENT, "Invalid UTF-8: \xFF", &error);

  XCTAssertNotNil(error);
  XCTAssertEqualObjects(error.domain, @"onnxruntime");
  XCTAssertEqual(error.code, ORT_INVALID_ARGUMENT);
  XCTAssertNil(error.userInfo[NSLocalizedDescriptionKey]);
}

- (void)testSaveExceptionWithInvalidUTF8Description {
  const std::runtime_error exception{"Invalid UTF-8: \xFF"};
  NSError* error = nil;
  ORTSaveExceptionToError(exception, &error);

  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, ORT_RUNTIME_EXCEPTION);
  XCTAssertNil(error.userInfo[NSLocalizedDescriptionKey]);
}

- (void)testSaveOrtExceptionWithInvalidUTF8Description {
  const Ort::Exception exception{"Invalid UTF-8: \xFF", ORT_INVALID_ARGUMENT};
  NSError* error = nil;
  ORTSaveOrtExceptionToError(exception, &error);

  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, ORT_INVALID_ARGUMENT);
  XCTAssertNil(error.userInfo[NSLocalizedDescriptionKey]);
}

- (void)testSaveDescriptionWithoutErrorOutput {
  ORTSaveCodeAndDescriptionToError(ORT_FAIL, "An error", nullptr);
  ORTSaveCodeAndDescriptionToError(ORT_FAIL, "Invalid UTF-8: \xFF", nullptr);
  ORTSaveCodeAndDescriptionToError(ORT_FAIL, @"An error", nullptr);
}

@end

NS_ASSUME_NONNULL_END
