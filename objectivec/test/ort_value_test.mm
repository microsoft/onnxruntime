// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

#import "ort_value.h"

#include <vector>

#import "test/assertion_utils.h"

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
  ORTValue* ortValue = [[ORTValue alloc] initWithTensorData:data
                                                elementType:elementType
                                                      shape:shape
                                                      error:&err];
  ORTAssertNullableResultSuccessful(ortValue, err);

  auto checkTensorInfo = [&](ORTTensorTypeAndShapeInfo* tensorInfo) {
    XCTAssertEqual(tensorInfo.elementType, elementType);
    XCTAssertEqualObjects(tensorInfo.shape, shape);
  };

  ORTValueTypeInfo* typeInfo = [ortValue typeInfoWithError:&err];
  ORTAssertNullableResultSuccessful(typeInfo, err);
  XCTAssertEqual(typeInfo.type, ORTValueTypeTensor);
  XCTAssertNotNil(typeInfo.tensorTypeAndShapeInfo);
  checkTensorInfo(typeInfo.tensorTypeAndShapeInfo);

  ORTTensorTypeAndShapeInfo* tensorInfo = [ortValue tensorTypeAndShapeInfoWithError:&err];
  ORTAssertNullableResultSuccessful(tensorInfo, err);
  checkTensorInfo(tensorInfo);

  NSData* actualData = [ortValue tensorDataWithError:&err];
  ORTAssertNullableResultSuccessful(actualData, err);
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
  ORTValue* ortValue = [[ORTValue alloc] initWithTensorData:data
                                                elementType:ORTTensorElementDataTypeInt32
                                                      shape:shape
                                                      error:&err];
  ORTAssertNullableResultUnsuccessful(ortValue, err);
}

- (void)testInitTensorWithStringDataSucceeds {
  NSArray<NSString*>* stringData = @[ @"ONNX Runtime", @"is", @"the", @"best", @"AI", @"Framework" ];
  NSError* err = nil;
  ORTValue* stringValue = [[ORTValue alloc] initWithTensorStringData:stringData shape:@[ @3, @2 ] error:&err];
  ORTAssertNullableResultSuccessful(stringValue, err);

  NSArray<NSString*>* returnedStringData = [stringValue tensorStringDataWithError:&err];
  ORTAssertNullableResultSuccessful(returnedStringData, err);

  XCTAssertEqual([stringData count], [returnedStringData count]);
  XCTAssertTrue([stringData isEqualToArray:returnedStringData]);
}

@end

NS_ASSUME_NONNULL_END
