// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

#import "ort_enums.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * The ORT environment.
 */
@interface ORTEnv : NSObject

- (instancetype)init NS_UNAVAILABLE;

/**
 * Creates an ORT Environment.
 *
 * @param loggingLevel The environment logging level.
 * @param[out] error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 */
- (nullable instancetype)initWithLoggingLevel:(ORTLoggingLevel)loggingLevel
                                        error:(NSError**)error NS_DESIGNATED_INITIALIZER;

@end

NS_ASSUME_NONNULL_END
