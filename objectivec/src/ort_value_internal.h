// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_value.h"

#import "src/cxx_api.h"

NS_ASSUME_NONNULL_BEGIN

@interface ORTValue ()

- (nullable instancetype)initWithCAPIOrtValue:(OrtValue*)CAPIOrtValue
                           externalTensorData:(nullable NSMutableData*)externalTensorData
                                        error:(NSError**)error NS_DESIGNATED_INITIALIZER;

- (Ort::Value&)CXXAPIOrtValue;

@end

NS_ASSUME_NONNULL_END
