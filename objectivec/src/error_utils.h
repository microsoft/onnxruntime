// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

#include "core/session/onnxruntime_cxx_api.h"

NS_ASSUME_NONNULL_BEGIN

void ORTSaveCodeAndDescriptionToError(int code, const char* description, NSError** error);
void ORTSaveExceptionToError(const Ort::Exception& e, NSError** error);

// API implementation wrapper macros to handle ORT C++ API exceptions
// clang-format off
#define ORT_OBJC_API_IMPL_BEGIN \
  try {

#define ORT_OBJC_API_IMPL_END(error, failure_return_value) \
  } catch (const Ort::Exception& e) {                      \
    ORTSaveExceptionToError(e, (error));                   \
    return (failure_return_value);                         \
  }
// clang-format on

#define ORT_OBJC_API_IMPL_END_BOOL(error) \
  ORT_OBJC_API_IMPL_END(error, NO)

#define ORT_OBJC_API_IMPL_END_NULLABLE(error) \
  ORT_OBJC_API_IMPL_END(error, nil)

NS_ASSUME_NONNULL_END
