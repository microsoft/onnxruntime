// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "onnxruntime/ort_env.h"

#include "core/session/onnxruntime_cxx_api.h"

NS_ASSUME_NONNULL_BEGIN

@interface ORTEnv ()

- (Ort::Env*)internalORTEnv;

@end

NS_ASSUME_NONNULL_END
