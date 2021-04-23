// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

#import "onnxruntime/ort_value.h"

#include <vector>

NS_ASSUME_NONNULL_BEGIN

@interface ORTValueTest : XCTestCase
@end

@implementation ORTValueTest

- (void)setUp {
  self.continueAfterFailure = NO;
}

- (void)testInitTensorOk {
  int32_t value = 42;
  NSMutableData* data = [[NSMutableData alloc] initWithBytes:&value
                                                      length:sizeof(int32_t)];
  NSArray<NSNumber*>* shape = @[ @1 ];

  NSError* err = nil;
  ORTValue* ortValue = [[ORTValue alloc] initTensorWithData:data
                                                elementType:ORTTensorElementDataTypeInt32
                                                      shape:shape
                                                      error:&err];
  XCTAssertNotNil(ortValue);
  XCTAssertNil(err);

  ORTValueType actualValueType;
  XCTAssertTrue([ortValue valueType:&actualValueType error:&err]);
  XCTAssertNil(err);
  XCTAssertEqual(actualValueType, ORTValueTypeTensor);

  ORTTensorElementDataType actualElementType;
  XCTAssertTrue([ortValue tensorElementType:&actualElementType error:&err]);
  XCTAssertNil(err);
  XCTAssertEqual(actualElementType, ORTTensorElementDataTypeInt32);

  NSArray<NSNumber*>* actualShape = [ortValue tensorShapeWithError:&err];
  XCTAssertNotNil(actualShape);
  XCTAssertNil(err);
  XCTAssertEqualObjects(shape, actualShape);

  NSData* actualData = [ortValue tensorDataWithError:&err];
  XCTAssertNotNil(actualData);
  XCTAssertNil(err);
  XCTAssertEqual(actualData.length, sizeof(int32_t));
  int32_t actualValue;
  memcpy(&actualValue, actualData.bytes, sizeof(int32_t));
  XCTAssertEqual(actualValue, value);
}

- (void)testInitTensorFailsWithDataSmallerThanShape {
  std::vector<int32_t> values{1, 2, 3, 4};
  NSMutableData* data = [[NSMutableData alloc] initWithBytes:values.data()
                                                      length:values.size() * sizeof(int32_t)];
  NSArray<NSNumber*>* shape = @[ @2, @3 ];  // too large

  NSError* err = nil;
  ORTValue* ort_value = [[ORTValue alloc] initTensorWithData:data
                                                 elementType:ORTTensorElementDataTypeInt32
                                                       shape:shape
                                                       error:&err];
  XCTAssertNil(ort_value);
  XCTAssertNotNil(err);
}

@end

NS_ASSUME_NONNULL_END
