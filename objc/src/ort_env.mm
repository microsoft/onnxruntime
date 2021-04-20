// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "onnxruntime/ort_env.h"
#import "src/ort_env_internal.h"
#import "src/error_utils.h"

#include <optional>

#include "core/session/onnxruntime_cxx_api.h"

NS_ASSUME_NONNULL_BEGIN

@implementation ORTEnv {
  std::optional<Ort::Env> _env;
}

- (nullable instancetype)initWithError:(NSError**)error {
  self = [super init];
  if (self) {
    try {
      _env = Ort::Env{};
    } catch (const Ort::Exception& e) {
      [ORTErrorUtils saveErrorCode:e.GetOrtErrorCode()
                       description:e.what()
                           toError:error];
      self = nil;
    }
  }
  return self;
}

- (Ort::Env*)handle {
  return &(*_env);
}

@end

NS_ASSUME_NONNULL_END
