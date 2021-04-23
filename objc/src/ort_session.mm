// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "onnxruntime/ort_session.h"

#include <optional>
#include <vector>

#include "core/session/onnxruntime_cxx_api.h"

#import "src/error_utils.h"
#import "src/ort_env_internal.h"
#import "src/ort_value_internal.h"

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
      _session = Ort::Session{*[env CXXAPIOrtEnv], path.UTF8String, sessionOptions};
    } catch (const Ort::Exception& e) {
      ORTSaveExceptionToError(e, error);
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
      inputValues.push_back(static_cast<const OrtValue*>(*[inputs[inputName] CXXAPIOrtValue]));
    }

    for (NSString* outputName in outputs) {
      outputNames.push_back(outputName.UTF8String);
      outputValues.push_back(static_cast<OrtValue*>(*[outputs[outputName] CXXAPIOrtValue]));
    }

    Ort::ThrowOnError(Ort::GetApi().Run(*_session, runOptions,
                                        inputNames.data(), inputValues.data(), inputNames.size(),
                                        outputNames.data(), outputNames.size(), outputValues.data()));

    status = YES;
  } catch (const Ort::Exception& e) {
    ORTSaveExceptionToError(e, error);
  }
  return status;
}

- (nullable NSDictionary<NSString*, ORTValue*>*)runWithInputs:(NSDictionary<NSString*, ORTValue*>*)inputs
                                                  outputNames:(NSSet<NSString*>*)outputNameSet
                                                        error:(NSError**)error {
  try {
    NSArray<NSString*>* outputNameArray = outputNameSet.allObjects;

    Ort::RunOptions runOptions{};  // TODO make configurable

    std::vector<const char*> inputNames, outputNames;
    std::vector<const OrtValue*> inputValues;
    std::vector<OrtValue*> outputValues;

    for (NSString* inputName in inputs) {
      inputNames.push_back(inputName.UTF8String);
      inputValues.push_back(static_cast<const OrtValue*>(*[inputs[inputName] CXXAPIOrtValue]));
    }

    for (NSString* outputName in outputNameArray) {
      outputNames.push_back(outputName.UTF8String);
      outputValues.push_back(nullptr);
    }

    Ort::ThrowOnError(Ort::GetApi().Run(*_session, runOptions,
                                        inputNames.data(), inputValues.data(), inputNames.size(),
                                        outputNames.data(), outputNames.size(), outputValues.data()));

    NSMutableDictionary<NSString*, ORTValue*>* outputs = [[NSMutableDictionary alloc] init];
    for (NSUInteger i = 0; i < outputNameArray.count; ++i) {
      ORTValue* outputValue = [[ORTValue alloc] initWithCAPIOrtValue:outputValues[i] externalTensorData:nil error:error];
      if (!outputValue) {
        // clean up remaining C API OrtValues which haven't been wrapped by an ORTValue yet
        for (NSUInteger j = i; j < outputNameArray.count; ++j) {
          Ort::GetApi().ReleaseValue(outputValues[j]);
        }
        return nil;
      }

      outputs[outputNameArray[i]] = outputValue;
    }

    return outputs;
  } catch (const Ort::Exception& e) {
    ORTSaveExceptionToError(e, error);
    return nil;
  }
}

@end

NS_ASSUME_NONNULL_END
