// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

#import "onnxruntime/ort_env.h"
#import "onnxruntime/ort_session.h"
#import "onnxruntime/ort_value.h"

@interface ORTSessionTest : XCTestCase
@end

static NSString* kTestDataPath = [NSString stringWithFormat: @"%@/Contents/Resources/testdata",
                                  [[NSBundle bundleForClass:[ORTSessionTest class]] bundlePath]];

@implementation ORTSessionTest

+ (NSMutableData*)dataWithScalarFloat:(float)value {
    NSMutableData* data = [[NSMutableData alloc] initWithBytes:&value length:sizeof(value)];
    return data;
}

+ (ORTValue*)ortValueWithScalarFloatData:(NSMutableData*)data {
    const int64_t first_dim = 1;
    const int64_t* shape = &first_dim;
    const size_t shape_len = 1;
    ORTValue* ort_value = [[ORTValue alloc] initTensorWithData:data
                                                   elementType:ORTElementDataTypeFloat
                                                         shape:shape
                                                      shapeLen:shape_len
                                                         error:nil];
    XCTAssertNotNil(ort_value);
    return ort_value;
}

- (void)testRunModel {
    // inputs: A, B
    // output: C = A + B
    NSString* add_model_path = [kTestDataPath stringByAppendingString:@"/single_add.onnx"];

    ORTEnv* env = [[ORTEnv alloc] initWithError:nil];
    XCTAssertNotNil(env);

    NSMutableData* a_data = [ORTSessionTest dataWithScalarFloat:1.0f];
    NSMutableData* b_data = [ORTSessionTest dataWithScalarFloat:2.0f];
    NSMutableData* c_data = [ORTSessionTest dataWithScalarFloat:0.0f];

    ORTValue* a = [ORTSessionTest ortValueWithScalarFloatData:a_data];
    ORTValue* b = [ORTSessionTest ortValueWithScalarFloatData:b_data];
    ORTValue* c = [ORTSessionTest ortValueWithScalarFloatData:c_data];

    ORTSession* session = [[ORTSession alloc] initWithEnv:env
                                                modelPath:add_model_path
                                                    error:nil];
    XCTAssertNotNil(session);

    BOOL result = [session runWithInputs:@{@"A": a, @"B": b}
                                 outputs:@{@"C": c}
                                   error:nil];
    XCTAssertTrue(result);

    const float c_expected = 3.0f;
    float c_actual;
    memcpy(&c_actual, c_data.bytes, sizeof(float));

    XCTAssertEqual(c_actual, c_expected);
}

@end
