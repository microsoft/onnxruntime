// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_APIS
#import "ort_training_session_internal.h"

#import <vector>
#import <optional>
#import <string>

#import "cxx_api.h"
#import "cxx_utils.h"
#import "error_utils.h"
#import "ort_checkpoint_internal.h"
#import "ort_session_internal.h"
#import "ort_enums_internal.h"
#import "ort_env_internal.h"
#import "ort_value_internal.h"

NS_ASSUME_NONNULL_BEGIN

@implementation ORTTrainingSession {
  std::optional<Ort::TrainingSession> _session;
}

- (Ort::TrainingSession&)CXXAPIOrtTrainingSession {
  return *_session;
}

- (nullable instancetype)initWithEnv:(ORTEnv*)env
                      sessionOptions:(ORTSessionOptions*)sessionOptions
                          checkPoint:(ORTCheckpoint*)checkPoint
                      trainModelPath:(NSString*)trainModelPath
                       evalModelPath:(nullable NSString*)evalModelPath
                  optimizerModelPath:(nullable NSString*)optimizerModelPath
                               error:(NSError**)error {
  if ((self = [super init]) == nil) {
    return nil;
  }

  try {
    std::optional<std::string> evalPath = Utils::toStdOptionalString(evalModelPath);
    std::optional<std::string> optimizerPath = Utils::toStdOptionalString(optimizerModelPath);

    _session = Ort::TrainingSession{
        [env CXXAPIOrtEnv],
        [sessionOptions CXXAPIOrtSessionOptions],
        [checkPoint CXXAPIOrtCheckpoint],
        trainModelPath.UTF8String,
        evalPath,
        optimizerPath};
    return self;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable NSArray<ORTValue*>*)trainStepWithInputValues:(NSArray<ORTValue*>*)inputs
                                                   error:(NSError**)error {
  try {
    std::vector<const OrtValue*> inputValues = Utils::toOrtValueVector(inputs);

    size_t outputCount;
    Ort::ThrowOnError(Ort::GetTrainingApi().TrainingSessionGetTrainingModelOutputCount(*_session, &outputCount));
    std::vector<OrtValue*> outputValues(outputCount, nullptr);

    Ort::RunOptions runOptions;
    Ort::ThrowOnError(Ort::GetTrainingApi().TrainStep(
        *_session,
        runOptions,
        inputValues.size(),
        inputValues.data(),
        outputValues.size(),
        outputValues.data()));

    return Utils::toORTValueNSArray(outputValues, error);
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}
- (nullable NSArray<ORTValue*>*)evalStepWithInputValues:(NSArray<ORTValue*>*)inputs
                                                  error:(NSError**)error {
  try {
    // create vector of Ort::Value from NSArray<ORTValue*> with same size as inputValues
    std::vector<const OrtValue*> inputValues = Utils::toOrtValueVector(inputs);

    size_t outputCount;
    Ort::ThrowOnError(Ort::GetTrainingApi().TrainingSessionGetEvalModelOutputCount(*_session, &outputCount));
    std::vector<OrtValue*> outputValues(outputCount, nullptr);

    Ort::RunOptions runOptions;
    Ort::ThrowOnError(Ort::GetTrainingApi().EvalStep(
        *_session,
        runOptions,
        inputValues.size(),
        inputValues.data(),
        outputValues.size(),
        outputValues.data()));

    return Utils::toORTValueNSArray(outputValues, error);
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (BOOL)lazyResetGradWithError:(NSError**)error {
  try {
    [self CXXAPIOrtTrainingSession].LazyResetGrad();
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)optimizerStepWithError:(NSError**)error {
  try {
    [self CXXAPIOrtTrainingSession].OptimizerStep();
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (nullable NSArray<NSString*>*)getTrainInputNamesWithError:(NSError**)error {
  try {
    std::vector<std::string> inputNames = [self CXXAPIOrtTrainingSession].InputNames(true);
    return Utils::toNSStringNSArray(inputNames);
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable NSArray<NSString*>*)getTrainOutputNamesWithError:(NSError**)error {
  try {
    std::vector<std::string> outputNames = [self CXXAPIOrtTrainingSession].OutputNames(true);
    return Utils::toNSStringNSArray(outputNames);
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable NSArray<NSString*>*)getEvalInputNamesWithError:(NSError**)error {
  try {
    std::vector<std::string> inputNames = [self CXXAPIOrtTrainingSession].InputNames(false);
    return Utils::toNSStringNSArray(inputNames);
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable NSArray<NSString*>*)getEvalOutputNamesWithError:(NSError**)error {
  try {
    std::vector<std::string> outputNames = [self CXXAPIOrtTrainingSession].OutputNames(false);
    return Utils::toNSStringNSArray(outputNames);
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (BOOL)registerLinearLRSchedulerWithWarmupStepCount:(int64_t)warmupStepCount
                                      totalStepCount:(int64_t)totalStepCount
                                           initialLr:(float)initialLr
                                               error:(NSError**)error {
  try {
    [self CXXAPIOrtTrainingSession].RegisterLinearLRScheduler(warmupStepCount, totalStepCount, initialLr);
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)schedulerStepWithError:(NSError**)error {
  try {
    [self CXXAPIOrtTrainingSession].SchedulerStep();
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (float)getLearningRateWithError:(NSError**)error {
  try {
    return [self CXXAPIOrtTrainingSession].GetLearningRate();
  }
  ORT_OBJC_API_IMPL_CATCH(error, 0.0f);
}

- (BOOL)setLearningRate:(float)lr
                  error:(NSError**)error {
  try {
    [self CXXAPIOrtTrainingSession].SetLearningRate(lr);
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)fromBufferWithValue:(ORTValue*)buffer
                      error:(NSError**)error {
  try {
    [self CXXAPIOrtTrainingSession].FromBuffer([buffer CXXAPIOrtValue]);
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (nullable ORTValue*)toBufferWithTrainable:(BOOL)onlyTrainable
                                      error:(NSError**)error {
  try {
    Ort::Value val = [self CXXAPIOrtTrainingSession].ToBuffer(onlyTrainable);
    return [[ORTValue alloc] initWithCAPIOrtValue:val.release()
                               externalTensorData:nil
                                            error:error];
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (BOOL)exportModelForInferenceWithOutputPath:(NSString*)infernceModelPath
                             graphOutputNames:(NSArray<NSString*>*)graphOutputNames
                                        error:(NSError**)error {
  try {
    [self CXXAPIOrtTrainingSession].ExportModelForInferencing(Utils::toStdString(infernceModelPath),
                                                              Utils::toStdStringVector(graphOutputNames));
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

@end

void ORTSetSeed(int64_t seed) {
  Ort::SetSeed(seed);
}

NS_ASSUME_NONNULL_END

#endif  // ENABLE_TRAINING_APIS
