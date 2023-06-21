// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>

#import "ort_session.h"
#import "ort_env.h"
#import "ort_value.h"

NS_ASSUME_NONNULL_BEGIN

namespace testUtils {

NSString* _Nullable createTemporaryDirectory(XCTestCase* testCase);

NSArray<NSNumber*>* getFloatArrayFromData(NSData* data);

}  // namespace testUtils

NS_ASSUME_NONNULL_END
