// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>
#include <stdint.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * An ORT checkpoint is a snapshot of the state of a model at a given point in time.
 *
 * This class holds the entire training session state that includes model parameters,
 * their gradients, optimizer parameters, and user properties. The `ORTTrainingSession` leverages the
 * `ORTCheckpoint` by accessing and updating the contained training state.
 *
 * Available since 1.16.
 *
 * @note This class is only available when the training APIs are enabled.
 */
@interface ORTCheckpoint : NSObject

- (instancetype)init NS_UNAVAILABLE;

/**
 * Creates a checkpoint from directory on disk.
 *
 * @param path The path to the checkpoint directory.
 * @param error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 *
 * @warning The construction of the checkpoint state requires instantiation of `ORTEnv`.
 * The intialization will fail if the `ORTEnv` is not properly initialized.
 */
- (nullable instancetype)initWithPath:(NSString*)path
                                error:(NSError**)error NS_DESIGNATED_INITIALIZER;

/**
 * Saves a checkpoint to directory on disk.
 *
 * @param path The path to the checkpoint directory.
 * @param includeOptimizerState Flag to indicate whether to save the optimizer state or not.
 * @param error Optional error information set if an error occurs.
 * @return Whether the checkpoint was saved successfully.
 */
- (BOOL)saveCheckpointToPath:(NSString*)path
          withOptimizerState:(BOOL)includeOptimizerState
                       error:(NSError**)error;

/**
 * Adds an int property to this checkpoint.
 *
 * @param name The name of the property.
 * @param value The value of the property.
 * @param error Optional error information set if an error occurs.
 * @return Whether the property was added successfully.
 */
- (BOOL)addIntPropertyWithName:(NSString*)name
                         value:(int64_t)value
                         error:(NSError**)error;

/**
 * Adds a float property to this checkpoint.
 *
 * @param name The name of the property.
 * @param value The value of the property.
 * @param error Optional error information set if an error occurs.
 * @return Whether the property was added successfully.
 */
- (BOOL)addFloatPropertyWithName:(NSString*)name
                           value:(float)value
                           error:(NSError**)error;

/**
 * Adds a string property to this checkpoint.
 *
 * @param name The name of the property.
 * @param value The value of the property.
 * @param error Optional error information set if an error occurs.
 * @return Whether the property was added successfully.
 */

- (BOOL)addStringPropertyWithName:(NSString*)name
                            value:(NSString*)value
                            error:(NSError**)error;

/**
 * Gets an int property from this checkpoint.
 *
 * @param name The name of the property.
 * @param error Optional error information set if an error occurs.
 * @return The value of the property or 0 if an error occurs.
 */
- (int64_t)getIntPropertyWithName:(NSString*)name
                            error:(NSError**)error __attribute__((swift_error(nonnull_error)));

/**
 * Gets a float property from this checkpoint.
 *
 * @param name The name of the property.
 * @param error Optional error information set if an error occurs.
 * @return The value of the property or 0.0f if an error occurs.
 */
- (float)getFloatPropertyWithName:(NSString*)name
                            error:(NSError**)error __attribute__((swift_error(nonnull_error)));

/**
 *
 * Gets a string property from this checkpoint.
 *
 * @param name The name of the property.
 * @param error Optional error information set if an error occurs.
 * @return The value of the property.
 */
- (nullable NSString*)getStringPropertyWithName:(NSString*)name
                                          error:(NSError**)error __attribute__((swift_error(nonnull_error)));

@end

NS_ASSUME_NONNULL_END
