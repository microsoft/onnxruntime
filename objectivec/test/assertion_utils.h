// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

NS_ASSUME_NONNULL_BEGIN

#define ORTAssertNullableResultSuccessful(result, error)                               \
  do {                                                                                 \
    XCTAssertNotNil(result, @"Expected non-nil result but got nil. Error: %@", error); \
    XCTAssertNil(error);                                                               \
  } while (0)

#define ORTAssertBoolResultSuccessful(result, error)                                \
  do {                                                                              \
    XCTAssertTrue(result, @"Expected true result but got false. Error: %@", error); \
    XCTAssertNil(error);                                                            \
  } while (0)

#define ORTAssertNullableResultUnsuccessful(result, error) \
  do {                                                     \
    XCTAssertNil(result);                                  \
    XCTAssertNotNil(error);                                \
  } while (0)

#define ORTAssertBoolResultUnsuccessful(result, error) \
  do {                                                 \
    XCTAssertFalse(result);                            \
    XCTAssertNotNil(error);                            \
  } while (0)

NS_ASSUME_NONNULL_END
