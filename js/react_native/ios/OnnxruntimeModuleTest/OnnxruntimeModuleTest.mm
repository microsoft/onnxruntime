// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>
#import "onnxruntime_cxx_api.h"
#import "OnnxruntimeModule.h"
#import "TensorHelper.h"

@interface OnnxruntimeModuleTest : XCTestCase

@end

@implementation OnnxruntimeModuleTest

- (void)testOnnxruntimeModule {
  NSBundle* bundle = [NSBundle bundleForClass:[OnnxruntimeModuleTest class]];
  NSString* dataPath = [bundle pathForResource:@"test_types_FLOAT" ofType:@"pb"];

  OnnxruntimeModule* onnxruntimeModule = [OnnxruntimeModule new];
  
  // test loadModel()
  {
    NSMutableDictionary* options = [NSMutableDictionary dictionary];
    NSDictionary* resultMap = [onnxruntimeModule loadModel:dataPath options:options];
    
    XCTAssertEqual(resultMap[@"key"], dataPath);
    NSArray* inputNames = resultMap[@"inputNames"];
    XCTAssertEqual([inputNames count], 1);
    XCTAssertEqualObjects(inputNames[0], @"input");
    NSArray* outputNames = resultMap[@"outputNames"];
    XCTAssertEqual([outputNames count], 1);
    XCTAssertEqualObjects(outputNames[0], @"output");
  }
  
  // test run()
  {
    NSMutableDictionary* inputTensorMap = [NSMutableDictionary dictionary];
    
    // dims
    NSArray* dims = @[[NSNumber numberWithLong:1],
                      [NSNumber numberWithLong:5]];
    inputTensorMap[@"dims"] = dims;

    // type
    inputTensorMap[@"type"] = JsTensorTypeFloat;

    // data
    std::array<float_t, 5> outValues{std::numeric_limits<float_t>::min(), 1.0f, -2.0f, 3.0f, std::numeric_limits<float_t>::max()};
    
    const NSInteger byteBufferSize = outValues.size() * sizeof(float_t);
    unsigned char* byteBuffer = static_cast<unsigned char*>(malloc(byteBufferSize));
    NSData* byteBufferRef = [NSData dataWithBytesNoCopy:byteBuffer length:byteBufferSize];
    float* floatPtr = (float*)[byteBufferRef bytes];
    for (NSUInteger i = 0; i < outValues.size(); ++i) {
      *floatPtr++ = outValues[i];
    }
    floatPtr = (float*)[byteBufferRef bytes];
 
    NSString* dataEncoded = [byteBufferRef base64EncodedStringWithOptions:0];
    inputTensorMap[@"data"] = dataEncoded;
    
    NSMutableDictionary* inputDataMap = [NSMutableDictionary dictionary];
    inputDataMap[@"input"] = inputTensorMap;
    
    NSMutableDictionary* options = [NSMutableDictionary dictionary];
    
    NSMutableArray* output = [NSMutableArray array];
    [output addObject:@"output"];

    NSDictionary* resultMap = [onnxruntimeModule run:dataPath input:inputDataMap output:output options:options];
    
    XCTAssertTrue([[resultMap objectForKey:@"output"] isEqualToDictionary:inputTensorMap]);
  }
}

@end
