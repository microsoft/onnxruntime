// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_value.h"

#import "cxx_api.h"

NS_ASSUME_NONNULL_BEGIN

@interface ORTValue ()

/**
 * Creates a value that wraps an existing C API OrtValue and takes ownership of it.
 *
 * @param CAPIOrtValue The C API OrtValue to wrap.
 * @param externalTensorData Any external tensor data referenced by `CAPIOrtValue`.
 * @param error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 */
- (nullable instancetype)initWithCAPIOrtValue:(OrtValue*)CAPIOrtValue
                           externalTensorData:(nullable NSMutableData*)externalTensorData
                                        error:(NSError**)error NS_DESIGNATED_INITIALIZER;

- (Ort::Value&)CXXAPIOrtValue;

@end

NS_ASSUME_NONNULL_END
