// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_env.h"

#include "onnxruntime_cxx_api.h"

NS_ASSUME_NONNULL_BEGIN

@interface ORTEnv ()

- (Ort::Env&)CXXAPIOrtEnv;

@end

NS_ASSUME_NONNULL_END
