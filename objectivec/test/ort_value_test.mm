// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

#import "ort_value.h"

#include <vector>

NS_ASSUME_NONNULL_BEGIN

@interface ORTValueTest : XCTestCase
@end

@implementation ORTValueTest

- (void)setUp {
  [super setUp];

  self.continueAfterFailure = NO;
}

- (void)testInitTensorOk {
  int32_t value = 42;
  NSMutableData* data = [[NSMutableData alloc] initWithBytes:&value
                                                      length:sizeof(int32_t)];
  NSArray<NSNumber*>* shape = @[ @1 ];

  const ORTTensorElementDataType elementType = ORTTensorElementDataTypeInt32;

  NSError* err = nil;
  ORTValue* ortValue = [[ORTValue alloc] initTensorWithData:data
                                                elementType:elementType
                                                      shape:shape
                                                      error:&err];
  XCTAssertNotNil(ortValue);
  XCTAssertNil(err);

  auto checkTensorInfo = [&](ORTTensorTypeAndShapeInfo* tensorInfo) {
    XCTAssertEqual(tensorInfo.elementType, elementType);
    XCTAssertEqualObjects(tensorInfo.shape, shape);
  };

  ORTValueTypeInfo* typeInfo = [ortValue typeInfoWithError:&err];
  XCTAssertNotNil(typeInfo);
  XCTAssertNil(err);
  XCTAssertEqual(typeInfo.type, ORTValueTypeTensor);
  XCTAssertNotNil(typeInfo.tensorTypeAndShapeInfo);
  checkTensorInfo(typeInfo.tensorTypeAndShapeInfo);

  ORTTensorTypeAndShapeInfo* tensorInfo = [ortValue tensorTypeAndShapeInfoWithError:&err];
  XCTAssertNotNil(tensorInfo);
  XCTAssertNil(err);
  checkTensorInfo(tensorInfo);

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
  ORTValue* ortValue = [[ORTValue alloc] initTensorWithData:data
                                                elementType:ORTTensorElementDataTypeInt32
                                                      shape:shape
                                                      error:&err];
  XCTAssertNil(ortValue);
  XCTAssertNotNil(err);
}

@end

NS_ASSUME_NONNULL_END
