// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "OnnxruntimeModule.h"
#import "FakeRCTBlobManager.h"
#import "TensorHelper.h"

#import <XCTest/XCTest.h>
#import <onnxruntime/onnxruntime_cxx_api.h>

@interface OnnxruntimeModuleTest : XCTestCase

@end

@implementation OnnxruntimeModuleTest

FakeRCTBlobManager *fakeBlobManager = nil;

+ (void)initialize {
  if (self == [OnnxruntimeModuleTest class]) {
    fakeBlobManager = [FakeRCTBlobManager new];
  }
}

- (void)testOnnxruntimeModule {
  NSBundle *bundle = [NSBundle bundleForClass:[OnnxruntimeModuleTest class]];
  NSString *dataPath = [bundle pathForResource:@"test_types_float" ofType:@"ort"];
  NSString *sessionKey = @"";
  NSString *sessionKey2 = @"";

  OnnxruntimeModule *onnxruntimeModule = [OnnxruntimeModule new];
  [onnxruntimeModule setBlobManager:fakeBlobManager];

  {
    // test loadModelFromBuffer()
    NSMutableDictionary *options = [NSMutableDictionary dictionary];
    NSData *fileData = [NSData dataWithContentsOfFile:dataPath];

    NSDictionary *resultMap = [onnxruntimeModule loadModelFromBuffer:fileData options:options];
    sessionKey = resultMap[@"key"];
    NSArray *inputNames = resultMap[@"inputNames"];
    XCTAssertEqual([inputNames count], 1);
    XCTAssertEqualObjects(inputNames[0], @"input");
    NSArray *outputNames = resultMap[@"outputNames"];
    XCTAssertEqual([outputNames count], 1);
    XCTAssertEqualObjects(outputNames[0], @"output");

    // test loadModel()
    NSDictionary *resultMap2 = [onnxruntimeModule loadModel:dataPath options:options];
    sessionKey2 = resultMap2[@"key"];
    NSArray *inputNames2 = resultMap2[@"inputNames"];
    XCTAssertEqual([inputNames2 count], 1);
    XCTAssertEqualObjects(inputNames2[0], @"input");
    NSArray *outputNames2 = resultMap2[@"outputNames"];
    XCTAssertEqual([outputNames2 count], 1);
    XCTAssertEqualObjects(outputNames2[0], @"output");
  }

  // test run()
  {
    NSMutableDictionary *inputTensorMap = [NSMutableDictionary dictionary];

    // dims
    NSArray *dims = @[ [NSNumber numberWithLong:1], [NSNumber numberWithLong:5] ];
    inputTensorMap[@"dims"] = dims;

    // type
    inputTensorMap[@"type"] = JsTensorTypeFloat;

    // data
    std::array<float, 5> outValues{std::numeric_limits<float>::min(), 1.0f, -2.0f, 3.0f,
                                   std::numeric_limits<float>::max()};

    const NSInteger byteBufferSize = outValues.size() * sizeof(float);
    unsigned char *byteBuffer = static_cast<unsigned char *>(malloc(byteBufferSize));
    NSData *byteBufferRef = [NSData dataWithBytesNoCopy:byteBuffer length:byteBufferSize];
    float *floatPtr = (float *)[byteBufferRef bytes];
    for (NSUInteger i = 0; i < outValues.size(); ++i) {
      *floatPtr++ = outValues[i];
    }
    floatPtr = (float *)[byteBufferRef bytes];

    XCTAssertNotNil(fakeBlobManager);
    inputTensorMap[@"data"] = [fakeBlobManager testCreateData:byteBufferRef];

    NSMutableDictionary *inputDataMap = [NSMutableDictionary dictionary];
    inputDataMap[@"input"] = inputTensorMap;

    NSMutableDictionary *options = [NSMutableDictionary dictionary];

    NSMutableArray *output = [NSMutableArray array];
    [output addObject:@"output"];

    NSDictionary *resultMap = [onnxruntimeModule run:sessionKey input:inputDataMap output:output options:options];
    NSDictionary *resultMap2 = [onnxruntimeModule run:sessionKey2 input:inputDataMap output:output options:options];

    // Compare output & input, but data.blobId is different
    // dims
    XCTAssertTrue([[resultMap objectForKey:@"output"][@"dims"] isEqualToArray:inputTensorMap[@"dims"]]);
    XCTAssertTrue([[resultMap2 objectForKey:@"output"][@"dims"] isEqualToArray:inputTensorMap[@"dims"]]);

    // type
    XCTAssertEqual([resultMap objectForKey:@"output"][@"type"], JsTensorTypeFloat);
    XCTAssertEqual([resultMap2 objectForKey:@"output"][@"type"], JsTensorTypeFloat);

    // data ({ blobId, offset, size })
    XCTAssertEqual([[resultMap objectForKey:@"output"][@"data"][@"offset"] longValue], 0);
    XCTAssertEqual([[resultMap2 objectForKey:@"output"][@"data"][@"size"] longValue], byteBufferSize);
  }

  // test dispose
  {
    [onnxruntimeModule dispose:sessionKey];
    [onnxruntimeModule dispose:sessionKey2];
  }
}

- (void)testOnnxruntimeModule_AppendCoreml {
  NSBundle *bundle = [NSBundle bundleForClass:[OnnxruntimeModuleTest class]];
  NSString *dataPath = [bundle pathForResource:@"test_types_float" ofType:@"ort"];
  NSString *sessionKey = @"";

  OnnxruntimeModule *onnxruntimeModule = [OnnxruntimeModule new];
  [onnxruntimeModule setBlobManager:fakeBlobManager];

  {
    // test loadModel() with coreml options
    NSMutableDictionary *options = [NSMutableDictionary dictionary];

    // register coreml ep options
    NSMutableArray *epList = [NSMutableArray array];
    [epList addObject:@"coreml"];
    NSArray *immutableEpList = [NSArray arrayWithArray:epList];
    [options setObject:immutableEpList forKey:@"executionProviders"];

    NSDictionary *resultMap = [onnxruntimeModule loadModel:dataPath options:options];

    sessionKey = resultMap[@"key"];
    NSArray *inputNames = resultMap[@"inputNames"];
    XCTAssertEqual([inputNames count], 1);
    XCTAssertEqualObjects(inputNames[0], @"input");
    NSArray *outputNames = resultMap[@"outputNames"];
    XCTAssertEqual([outputNames count], 1);
    XCTAssertEqualObjects(outputNames[0], @"output");
  }

  { [onnxruntimeModule dispose:sessionKey]; }
}

@end
