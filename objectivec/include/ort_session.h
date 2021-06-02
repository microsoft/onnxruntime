// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

#import "ort_enums.h"

NS_ASSUME_NONNULL_BEGIN

@class ORTEnv;
@class ORTRunOptions;
@class ORTSessionOptions;
@class ORTValue;

/**
 * An ORT session loads and runs a model.
 */
@interface ORTSession : NSObject

- (instancetype)init NS_UNAVAILABLE;

/**
 * Creates a session.
 *
 * @param env The ORT Environment instance.
 * @param path The path to the ONNX model.
 * @param sessionOptions Optional session configuration options.
 * @param[out] error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 */
- (nullable instancetype)initWithEnv:(ORTEnv*)env
                           modelPath:(NSString*)path
                      sessionOptions:(nullable ORTSessionOptions*)sessionOptions
                               error:(NSError**)error NS_DESIGNATED_INITIALIZER;

/**
 * Runs the model.
 * The inputs and outputs are pre-allocated.
 *
 * @param inputs Dictionary of input names to input ORT values.
 * @param outputs Dictionary of output names to output ORT values.
 * @param runOptions Optional run configuration options.
 * @param[out] error Optional error information set if an error occurs.
 * @return Whether the model was run successfully.
 */
- (BOOL)runWithInputs:(NSDictionary<NSString*, ORTValue*>*)inputs
              outputs:(NSDictionary<NSString*, ORTValue*>*)outputs
           runOptions:(nullable ORTRunOptions*)runOptions
                error:(NSError**)error;

/**
 * Runs the model.
 * The inputs are pre-allocated and the outputs are allocated by ORT.
 *
 * @param inputs Dictionary of input names to input ORT values.
 * @param outputNames Set of output names.
 * @param runOptions Optional run configuration options.
 * @param[out] error Optional error information set if an error occurs.
 * @return A dictionary of output names to output ORT values with the outputs
 *         requested in `outputNames`, or nil if an error occurs.
 */
- (nullable NSDictionary<NSString*, ORTValue*>*)runWithInputs:(NSDictionary<NSString*, ORTValue*>*)inputs
                                                  outputNames:(NSSet<NSString*>*)outputNames
                                                   runOptions:(nullable ORTRunOptions*)runOptions
                                                        error:(NSError**)error;

/**
 * Gets the model's input names.
 *
 * @param[out] error Optional error information set if an error occurs.
 * @return An array of input names, or nil if an error occurs.
 */
- (nullable NSArray<NSString*>*)inputNamesWithError:(NSError**)error;

/**
 * Gets the model's overridable initializer names.
 *
 * @param[out] error Optional error information set if an error occurs.
 * @return An array of overridable initializer names, or nil if an error occurs.
 */
- (nullable NSArray<NSString*>*)overridableInitializerNamesWithError:(NSError**)error;

/**
 * Gets the model's output names.
 *
 * @param[out] error Optional error information set if an error occurs.
 * @return An array of output names, or nil if an error occurs.
 */
- (nullable NSArray<NSString*>*)outputNamesWithError:(NSError**)error;

@end

/**
 * Options for configuring a session.
 */
@interface ORTSessionOptions : NSObject

- (instancetype)init NS_UNAVAILABLE;

/**
 * Creates session configuration options.
 *
 * @param[out] error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 */
- (nullable instancetype)initWithError:(NSError**)error NS_SWIFT_NAME(init());

/**
 * Sets the number of threads used to parallelize the execution within nodes.
 * A value of 0 means ORT will pick a default value.
 *
 * @param intraOpNumThreads The number of threads.
 * @param[out] error Optional error information set if an error occurs.
 * @return Whether the option was set successfully.
 */
- (BOOL)setIntraOpNumThreads:(int)intraOpNumThreads
                       error:(NSError**)error;

/**
 * Sets the graph optimization level.
 *
 * @param graphOptimizationLevel The graph optimization level.
 * @param[out] error Optional error information set if an error occurs.
 * @return Whether the option was set successfully.
 */
- (BOOL)setGraphOptimizationLevel:(ORTGraphOptimizationLevel)graphOptimizationLevel
                            error:(NSError**)error;

/**
 * Sets the path to which the optimized model file will be saved.
 *
 * @param optimizedModelFilePath The optimized model file path.
 * @param[out] error Optional error information set if an error occurs.
 * @return Whether the option was set successfully.
 */
- (BOOL)setOptimizedModelFilePath:(NSString*)optimizedModelFilePath
                            error:(NSError**)error;

/**
 * Sets the session log ID.
 *
 * @param logID The log ID.
 * @param[out] error Optional error information set if an error occurs.
 * @return Whether the option was set successfully.
 */
- (BOOL)setLogID:(NSString*)logID
           error:(NSError**)error;

/**
 * Sets the session log severity level.
 *
 * @param loggingLevel The log severity level.
 * @param[out] error Optional error information set if an error occurs.
 * @return Whether the option was set successfully.
 */
- (BOOL)setLogSeverityLevel:(ORTLoggingLevel)loggingLevel
                      error:(NSError**)error;

/**
 * Sets a session configuration key-value pair.
 * Any value for a previously set key will be overwritten.
 * The session configuration keys and values are documented here:
 * https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h
 *
 * @param key The key.
 * @param value The value.
 * @param[out] error Optional error information set if an error occurs.
 * @return Whether the option was set successfully.
 */
- (BOOL)addConfigEntryWithKey:(NSString*)key
                        value:(NSString*)value
                        error:(NSError**)error;

@end

/**
 * Options for configuring a run.
 */
@interface ORTRunOptions : NSObject

- (instancetype)init NS_UNAVAILABLE;

/**
 * Creates run configuration options.
 *
 * @param[out] error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 */
- (nullable instancetype)initWithError:(NSError**)error NS_SWIFT_NAME(init());

/**
 * Sets the run log tag.
 *
 * @param logTag The log tag.
 * @param[out] error Optional error information set if an error occurs.
 * @return Whether the option was set successfully.
 */
- (BOOL)setLogTag:(NSString*)logTag
            error:(NSError**)error;

/**
 * Sets the run log severity level.
 *
 * @param loggingLevel The log severity level.
 * @param[out] error Optional error information set if an error occurs.
 * @return Whether the option was set successfully.
 */
- (BOOL)setLogSeverityLevel:(ORTLoggingLevel)loggingLevel
                      error:(NSError**)error;

/**
 * Sets a run configuration key-value pair.
 * Any value for a previously set key will be overwritten.
 * The run configuration keys and values are documented here:
 * https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h
 *
 * @param key The key.
 * @param value The value.
 * @param[out] error Optional error information set if an error occurs.
 * @return Whether the option was set successfully.
 */
- (BOOL)addConfigEntryWithKey:(NSString*)key
                        value:(NSString*)value
                        error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
