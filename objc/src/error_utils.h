// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

#include "core/session/onnxruntime_cxx_api.h"

NS_ASSUME_NONNULL_BEGIN

void ORTThrowNotImplementedException(const char* description);
void ORTSaveCodeAndDescriptionToError(int code, const char* description, NSError** error);
void ORTSaveExceptionToError(const Ort::Exception& e, NSError** error);

NS_ASSUME_NONNULL_END
