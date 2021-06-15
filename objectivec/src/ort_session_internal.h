// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_session.h"

#include "onnxruntime_cxx_api.h"

NS_ASSUME_NONNULL_BEGIN

@interface ORTSessionOptions ()

- (Ort::SessionOptions&)CXXAPIOrtSessionOptions;

@end

@interface ORTRunOptions ()

- (Ort::RunOptions&)CXXAPIOrtRunOptions;

@end

NS_ASSUME_NONNULL_END
