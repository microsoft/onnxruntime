// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_training_session.h"

#import "cxx_api.h"

NS_ASSUME_NONNULL_BEGIN

@interface ORTTrainingSession ()

- (Ort::TrainingSession&)CXXAPIOrtTrainingSession;

@end

NS_ASSUME_NONNULL_END
