// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "onnxruntime/ort_value.h"

#include "core/session/onnxruntime_cxx_api.h"

@interface ORTValue ()
- (Ort::Value*)handle;
@end
