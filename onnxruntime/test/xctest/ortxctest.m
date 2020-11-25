// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>


// This is the stub test cases which will let the xcode command line tool start testing on Simulator 
@interface ONNXRuntimeTestXCWrapper : XCTestCase

@end

@implementation ONNXRuntimeTestXCWrapper

- (void)setUp {
    // Put setup code here. This method is called before the invocation of each test method in the class.
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
}

- (void)testExample {
    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
}

- (void)testPerformanceExample {
    // This is an example of a performance test case.
    [self measureBlock:^{
        // Put the code you want to measure the time of here.
    }];
}

@end
