// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "onnxruntime/ort_session.h"
#import "onnxruntime/ort_env.h"
#import "onnxruntime/ort_value.h"
#import "src/error_utils.h"
#import "src/ort_env_internal.h"
#import "src/ort_value_internal.h"

#include <vector>

#include "core/common/optional.h"
#include "core/session/onnxruntime_cxx_api.h"

NS_ASSUME_NONNULL_BEGIN

@implementation ORTSession {
  onnxruntime::optional<Ort::Session> _session;
}

- (nullable instancetype)initWithEnv:(ORTEnv*)env
                           modelPath:(NSString*)path
                               error:(NSError**)error {
  self = [super init];
  if (self) {
    try {
      Ort::Env* ort_env = [env handle];
      const char* path_cstr = path.UTF8String;
      Ort::SessionOptions session_options{};  // TODO make configurable
      _session = Ort::Session{*ort_env, path_cstr, session_options};
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
    Ort::RunOptions run_options{};  // TODO make configurable

    std::vector<const char*> input_names, output_names;
    std::vector<const OrtValue*> input_values;
    std::vector<OrtValue*> output_values;

    for (NSString* input_name in inputs) {
      input_names.push_back(input_name.UTF8String);
      input_values.push_back(static_cast<const OrtValue*>(*[inputs[input_name] handle]));
    }

    for (NSString* output_name in outputs) {
      output_names.push_back(output_name.UTF8String);
      output_values.push_back(static_cast<OrtValue*>(*[outputs[output_name] handle]));
    }

    Ort::ThrowOnError(Ort::GetApi().Run(*_session, run_options,
                                        input_names.data(), input_values.data(), input_names.size(),
                                        output_names.data(), output_names.size(), output_values.data()));

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
