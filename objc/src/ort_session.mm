// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "onnxruntime/ort_session.h"
#import "onnxruntime/ort_env.h"
#import "onnxruntime/ort_value.h"
#import "src/error_utils.h"
#import "src/ort_env_internal.h"
#import "src/ort_value_internal.h"

#include <optional>
#include <vector>

#include "core/session/onnxruntime_cxx_api.h"

NS_ASSUME_NONNULL_BEGIN

@implementation ORTSession {
  std::optional<Ort::Session> _session;
}

- (nullable instancetype)initWithEnv:(ORTEnv*)env
                           modelPath:(NSString*)path
                               error:(NSError**)error {
  self = [super init];
  if (self) {
    try {
      Ort::SessionOptions sessionOptions{};  // TODO make configurable
      _session = Ort::Session{*[env handle], path.UTF8String, sessionOptions};
    } catch (const Ort::Exception& e) {
      [ORTErrorUtils saveErrorCode:e.GetOrtErrorCode()
                       description:e.what()
                           toError:error];
      self = nil;
    }
  }
  return self;
}

- (BOOL)runWithInputs:(NSDictionary<NSString*, ORTValue*>*)inputs
              outputs:(NSDictionary<NSString*, ORTValue*>*)outputs
                error:(NSError**)error {
  BOOL status = NO;
  try {
    Ort::RunOptions runOptions{};  // TODO make configurable

    std::vector<const char*> inputNames, outputNames;
    std::vector<const OrtValue*> inputValues;
    std::vector<OrtValue*> outputValues;

    for (NSString* inputName in inputs) {
      inputNames.push_back(inputName.UTF8String);
      inputValues.push_back(static_cast<const OrtValue*>(*[inputs[inputName] handle]));
    }

    for (NSString* outputName in outputs) {
      outputNames.push_back(outputName.UTF8String);
      outputValues.push_back(static_cast<OrtValue*>(*[outputs[outputName] handle]));
    }

    Ort::ThrowOnError(Ort::GetApi().Run(*_session, runOptions,
                                        inputNames.data(), inputValues.data(), inputNames.size(),
                                        outputNames.data(), outputNames.size(), outputValues.data()));

    status = YES;
  } catch (const Ort::Exception& e) {
    [ORTErrorUtils saveErrorCode:e.GetOrtErrorCode()
                     description:e.what()
                         toError:error];
  }
  return status;
}

@end

NS_ASSUME_NONNULL_END
