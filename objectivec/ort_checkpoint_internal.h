// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_checkpoint.h"

#import "cxx_api.h"

NS_ASSUME_NONNULL_BEGIN

@interface ORTCheckpoint ()

- (Ort::CheckpointState&)CXXAPIOrtCheckpoint;

@end

NS_ASSUME_NONNULL_END
