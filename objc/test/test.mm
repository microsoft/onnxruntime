#import <XCTest/XCTest.h>

#import "onnxruntime/ort_env.h"

@interface ORTAPITest : XCTestCase
@end

@implementation ORTAPITest

- (void)testCreateEnv {
    NSError* error;
    ORTEnv* env = [[ORTEnv alloc] init:&error];
    XCTAssertNotNil(env);
    XCTAssertNil(error);
}

@end
