// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_session_internal.h"

#include <optional>
#include <vector>

#include "core/session/onnxruntime_cxx_api.h"

#import "src/error_utils.h"
#import "src/ort_enums_internal.h"
#import "src/ort_env_internal.h"
#import "src/ort_value_internal.h"

NS_ASSUME_NONNULL_BEGIN

@implementation ORTSession {
  std::optional<Ort::Session> _session;
}

- (nullable instancetype)initWithEnv:(ORTEnv*)env
                           modelPath:(NSString*)path
                      sessionOptions:(nullable ORTSessionOptions*)sessionOptions
                               error:(NSError**)error {
  if ((self = [super init]) == nil) {
    return nil;
  }

  try {
    if (!sessionOptions) {
      sessionOptions = [[ORTSessionOptions alloc] initWithError:error];
      if (!sessionOptions) {
        return nil;
      }
    }

    _session = Ort::Session{[env CXXAPIOrtEnv],
                            path.UTF8String,
                            [sessionOptions CXXAPIOrtSessionOptions]};

    return self;
  } catch (const Ort::Exception& e) {
    ORTSaveExceptionToError(e, error);
    return nil;
  }
}

- (BOOL)runWithInputs:(NSDictionary<NSString*, ORTValue*>*)inputs
              outputs:(NSDictionary<NSString*, ORTValue*>*)outputs
           runOptions:(nullable ORTRunOptions*)runOptions
                error:(NSError**)error {
  ORT_OBJC_API_IMPL_BEGIN

  if (!runOptions) {
    runOptions = [[ORTRunOptions alloc] initWithError:error];
    if (!runOptions) {
      return NO;
    }
  }

  std::vector<const char*> inputNames, outputNames;
  std::vector<const OrtValue*> inputValues;
  std::vector<OrtValue*> outputValues;

  for (NSString* inputName in inputs) {
    inputNames.push_back(inputName.UTF8String);
    inputValues.push_back(static_cast<const OrtValue*>([inputs[inputName] CXXAPIOrtValue]));
  }

  for (NSString* outputName in outputs) {
    outputNames.push_back(outputName.UTF8String);
    outputValues.push_back(static_cast<OrtValue*>([outputs[outputName] CXXAPIOrtValue]));
  }

  Ort::ThrowOnError(Ort::GetApi().Run(*_session, [runOptions CXXAPIOrtRunOptions],
                                      inputNames.data(), inputValues.data(), inputNames.size(),
                                      outputNames.data(), outputNames.size(), outputValues.data()));

  return YES;

  ORT_OBJC_API_IMPL_END_BOOL(error)
}

- (nullable NSDictionary<NSString*, ORTValue*>*)runWithInputs:(NSDictionary<NSString*, ORTValue*>*)inputs
                                                  outputNames:(NSSet<NSString*>*)outputNameSet
                                                   runOptions:(nullable ORTRunOptions*)runOptions
                                                        error:(NSError**)error {
  ORT_OBJC_API_IMPL_BEGIN

  if (!runOptions) {
    runOptions = [[ORTRunOptions alloc] initWithError:error];
    if (!runOptions) {
      return nil;
    }
  }

  NSArray<NSString*>* outputNameArray = outputNameSet.allObjects;

  std::vector<const char*> inputNames, outputNames;
  std::vector<const OrtValue*> inputValues;
  std::vector<OrtValue*> outputValues;

  for (NSString* inputName in inputs) {
    inputNames.push_back(inputName.UTF8String);
    inputValues.push_back(static_cast<const OrtValue*>([inputs[inputName] CXXAPIOrtValue]));
  }

  for (NSString* outputName in outputNameArray) {
    outputNames.push_back(outputName.UTF8String);
    outputValues.push_back(nullptr);
  }

  Ort::ThrowOnError(Ort::GetApi().Run(*_session, [runOptions CXXAPIOrtRunOptions],
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

  ORT_OBJC_API_IMPL_END_NULLABLE(error)
}

@end

@implementation ORTSessionOptions {
  std::optional<Ort::SessionOptions> _sessionOptions;
}

#pragma mark - Public

- (nullable instancetype)initWithError:(NSError**)error {
  self = [super init];
  if (self) {
    try {
      _sessionOptions = Ort::SessionOptions{};
    } catch (const Ort::Exception& e) {
      ORTSaveExceptionToError(e, error);
      self = nil;
    }
  }
  return self;
}

- (BOOL)setIntraOpNumThreads:(int)intraOpNumThreads
                       error:(NSError**)error {
  ORT_OBJC_API_IMPL_BEGIN
  _sessionOptions->SetIntraOpNumThreads(intraOpNumThreads);
  return YES;
  ORT_OBJC_API_IMPL_END_BOOL(error)
}

- (BOOL)setGraphOptimizationLevel:(ORTGraphOptimizationLevel)graphOptimizationLevel
                            error:(NSError**)error {
  ORT_OBJC_API_IMPL_BEGIN
  _sessionOptions->SetGraphOptimizationLevel(
      PublicToCAPIGraphOptimizationLevel(graphOptimizationLevel));
  return YES;
  ORT_OBJC_API_IMPL_END_BOOL(error)
}

- (BOOL)setOptimizedModelFilePath:(NSString*)optimizedModelFilePath
                            error:(NSError**)error {
  ORT_OBJC_API_IMPL_BEGIN
  _sessionOptions->SetOptimizedModelFilePath(optimizedModelFilePath.UTF8String);
  return YES;
  ORT_OBJC_API_IMPL_END_BOOL(error)
}

- (BOOL)setLogID:(NSString*)logID
           error:(NSError**)error {
  ORT_OBJC_API_IMPL_BEGIN
  _sessionOptions->SetLogId(logID.UTF8String);
  return YES;
  ORT_OBJC_API_IMPL_END_BOOL(error)
}

- (BOOL)setLogSeverityLevel:(ORTLoggingLevel)loggingLevel
                      error:(NSError**)error {
  ORT_OBJC_API_IMPL_BEGIN
  _sessionOptions->SetLogSeverityLevel(PublicToCAPILoggingLevel(loggingLevel));
  return YES;
  ORT_OBJC_API_IMPL_END_BOOL(error)
}

- (BOOL)addConfigEntryWithKey:(NSString*)key
                        value:(NSString*)value
                        error:(NSError**)error {
  ORT_OBJC_API_IMPL_BEGIN
  _sessionOptions->AddConfigEntry(key.UTF8String, value.UTF8String);
  return YES;
  ORT_OBJC_API_IMPL_END_BOOL(error)
}

#pragma mark - Internal

- (Ort::SessionOptions&)CXXAPIOrtSessionOptions {
  return *_sessionOptions;
}

@end

@implementation ORTRunOptions {
  std::optional<Ort::RunOptions> _runOptions;
}

#pragma mark - Public

- (nullable instancetype)initWithError:(NSError**)error {
  self = [super init];
  if (self) {
    try {
      _runOptions = Ort::RunOptions{};
    } catch (const Ort::Exception& e) {
      ORTSaveExceptionToError(e, error);
      self = nil;
    }
  }
  return self;
}

- (BOOL)setLogTag:(NSString*)logTag
            error:(NSError**)error {
  ORT_OBJC_API_IMPL_BEGIN
  _runOptions->SetRunTag(logTag.UTF8String);
  return YES;
  ORT_OBJC_API_IMPL_END_BOOL(error)
}

- (BOOL)setLogSeverityLevel:(ORTLoggingLevel)loggingLevel
                      error:(NSError**)error {
  ORT_OBJC_API_IMPL_BEGIN
  _runOptions->SetRunLogSeverityLevel(PublicToCAPILoggingLevel(loggingLevel));
  return YES;
  ORT_OBJC_API_IMPL_END_BOOL(error)
}

- (BOOL)addConfigEntryWithKey:(NSString*)key
                        value:(NSString*)value
                        error:(NSError**)error {
  ORT_OBJC_API_IMPL_BEGIN
  _runOptions->AddConfigEntry(key.UTF8String, value.UTF8String);
  return YES;
  ORT_OBJC_API_IMPL_END_BOOL(error)
}

#pragma mark - Internal

- (Ort::RunOptions&)CXXAPIOrtRunOptions {
  return *_runOptions;
}

@end

NS_ASSUME_NONNULL_END
