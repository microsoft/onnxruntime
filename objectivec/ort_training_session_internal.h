// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_APIS
#import "ort_training_session.h"

#import "cxx_api.h"

NS_ASSUME_NONNULL_BEGIN

@interface ORTTrainingSession ()

- (Ort::TrainingSession&)CXXAPIOrtTrainingSession;

@end

NS_ASSUME_NONNULL_END

#endif  // ENABLE_TRAINING_APIS
