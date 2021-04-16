// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

#import "onnxruntime/ort_value.h"

#include <vector>

@interface ORTValueTest : XCTestCase
@end

@implementation ORTValueTest

- (void)testInitOk {
  int32_t value = 42;
  NSMutableData* data = [[NSMutableData alloc] initWithBytes:&value
                                                      length:sizeof(int32_t)];
  const std::vector<int64_t> shape{1};

  NSError* err;
  ORTValue* ort_value = [[ORTValue alloc] initTensorWithData:data
                                                 elementType:ORTTensorElementDataTypeInt32
                                                       shape:shape.data()
                                                    shapeLen:shape.size()
                                                       error:&err];
  XCTAssertNotNil(ort_value);
  XCTAssertNil(err);
}

- (void)testInitFailsWithDataSmallerThanShape {
  std::vector<int32_t> values{1, 2, 3, 4};
  NSMutableData* data = [[NSMutableData alloc] initWithBytes:values.data()
                                                      length:values.size() * sizeof(int32_t)];
  const std::vector<int64_t> shape{2, 3};  // too large

  NSError* err;
  ORTValue* ort_value = [[ORTValue alloc] initTensorWithData:data
                                                 elementType:ORTTensorElementDataTypeInt32
                                                       shape:shape.data()
                                                    shapeLen:shape.size()
                                                       error:&err];
  XCTAssertNil(ort_value);
  XCTAssertNotNil(err);
}

@end
