// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class ORTEnv;
@class ORTValue;

/**
 * An ORT session loads and runs a model.
 */
@interface ORTSession : NSObject

- (nullable instancetype)init NS_UNAVAILABLE;

/**
 * Creates an ORT Session.
 *
 * @param env The ORT Environment instance.
 * @param path The path to the ONNX model.
 * @param[out] error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 */
- (nullable instancetype)initWithEnv:(ORTEnv*)env
                           modelPath:(NSString*)path
                               error:(NSError**)error NS_DESIGNATED_INITIALIZER;

/**
 * Runs the model.
 * The inputs and outputs are pre-allocated.
 *
 * @param inputs Dictionary of input names to input ORT values.
 * @param outputs Dictionary of output names to output ORT values.
 * @param[out] error Optional error information set if an error occurs.
 * @return Whether the model was run successfully.
 */
- (BOOL)runWithInputs:(NSDictionary<NSString*, ORTValue*>*)inputs
              outputs:(NSDictionary<NSString*, ORTValue*>*)outputs
                error:(NSError**)error;

/**
 * Runs the model.
 * The inputs are pre-allocated and the outputs are allocated by ORT.
 *
 * @param inputs Dictionary of input names to input ORT values.
 * @param outputNames Set of output names.
 * @param[out] error Optional error information set if an error occurs.
 * @return A dictionary of output names to output ORT values with the outputs
 *         requested in `outputNames`, or nil if an error occurs.
 */
- (nullable NSDictionary<NSString*, ORTValue*>*)runWithInputs:(NSDictionary<NSString*, ORTValue*>*)inputs
                                                  outputNames:(NSSet<NSString*>*)outputNames
                                                        error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
