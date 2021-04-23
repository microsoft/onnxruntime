// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "src/error_utils.h"

NS_ASSUME_NONNULL_BEGIN

static NSString* const kOrtErrorDomain = @"onnxruntime";

void ORTThrowNotImplementedException(const char* description) {
  throw Ort::Exception{description, ORT_NOT_IMPLEMENTED};
}

void ORTSaveCodeAndDescriptionToError(int code, const char* descriptionCstr, NSError** error) {
  if (!error) return;

  NSString* description = [NSString stringWithCString:descriptionCstr
                                             encoding:NSASCIIStringEncoding];

  *error = [NSError errorWithDomain:kOrtErrorDomain
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey : description}];
}

void ORTSaveExceptionToError(const Ort::Exception& e, NSError** error) {
  ORTSaveCodeAndDescriptionToError(e.GetOrtErrorCode(), e.what(), error);
}

NS_ASSUME_NONNULL_END
