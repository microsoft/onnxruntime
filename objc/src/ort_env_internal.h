// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "onnxruntime/ort_env.h"

#include "core/session/onnxruntime_cxx_api.h"

@interface ORTEnv ()
- (Ort::Env*)handle;
@end
