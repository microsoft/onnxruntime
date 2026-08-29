// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

#import "ort_custom_op_registration.h"
#import "ort_enums.h"

NS_ASSUME_NONNULL_BEGIN

@class ORTEnv;
@class ORTRunOptions;
@class ORTSessionOptions;
@class ORTValue;

/**
 * A block that supplies application-managed external EPContext data.
 *
 * Available since 1.30.
 *
 * The block may be called concurrently from arbitrary ONNX Runtime threads and must be thread-safe.
 *
 * @param name The logical name of the requested EPContext data.
 * @param error Error information to set when returning nil.
 * @return The requested data. Zero-length data is a successful empty payload. Return nil only on failure and set error.
 */
typedef NSData* _Nullable (^ORTEpContextDataReadBlock)(NSString* name, NSError** error);

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
 * @param error Optional error information set if an error occurs.
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
 * @param error Optional error information set if an error occurs.
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
 * @param error Optional error information set if an error occurs.
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
 * @param error Optional error information set if an error occurs.
 * @return An array of input names, or nil if an error occurs.
 */
- (nullable NSArray<NSString*>*)inputNamesWithError:(NSError**)error;

/**
 * Gets the model's overridable initializer names.
 *
 * @param error Optional error information set if an error occurs.
 * @return An array of overridable initializer names, or nil if an error occurs.
 */
- (nullable NSArray<NSString*>*)overridableInitializerNamesWithError:(NSError**)error;

/**
 * Gets the model's output names.
 *
 * @param error Optional error information set if an error occurs.
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
 * @param error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 */
- (nullable instancetype)initWithError:(NSError**)error NS_SWIFT_NAME(init());

/**
 * Appends an execution provider to the session options to enable the execution provider to be used when running
 * the model.
 *
 * Available since 1.14.
 *
 * The execution provider list is ordered by decreasing priority.
 * i.e. the first provider registered has the highest priority.
 *
 * @param providerName Provider name. For example, "xnnpack".
 * @param providerOptions Provider-specific options. For example, for provider "xnnpack", {"intra_op_num_threads": "2"}.
 * @param error Optional error information set if an error occurs.
 * @return Whether the execution provider was appended successfully
 */
- (BOOL)appendExecutionProvider:(NSString*)providerName
                providerOptions:(NSDictionary<NSString*, NSString*>*)providerOptions
                          error:(NSError**)error;
/**
 * Sets the number of threads used to parallelize the execution within nodes.
 * A value of 0 means ORT will pick a default value.
 *
 * @param intraOpNumThreads The number of threads.
 * @param error Optional error information set if an error occurs.
 * @return Whether the option was set successfully.
 */
- (BOOL)setIntraOpNumThreads:(int)intraOpNumThreads
                       error:(NSError**)error;

/**
 * Sets the graph optimization level.
 *
 * @param graphOptimizationLevel The graph optimization level.
 * @param error Optional error information set if an error occurs.
 * @return Whether the option was set successfully.
 */
- (BOOL)setGraphOptimizationLevel:(ORTGraphOptimizationLevel)graphOptimizationLevel
                            error:(NSError**)error;

/**
 * Sets the path to which the optimized model file will be saved.
 *
 * @param optimizedModelFilePath The optimized model file path.
 * @param error Optional error information set if an error occurs.
 * @return Whether the option was set successfully.
 */
- (BOOL)setOptimizedModelFilePath:(NSString*)optimizedModelFilePath
                            error:(NSError**)error;

/**
 * Sets the session log ID.
 *
 * @param logID The log ID.
 * @param error Optional error information set if an error occurs.
 * @return Whether the option was set successfully.
 */
- (BOOL)setLogID:(NSString*)logID
           error:(NSError**)error;

/**
 * Sets the session log severity level.
 *
 * @param loggingLevel The log severity level.
 * @param error Optional error information set if an error occurs.
 * @return Whether the option was set successfully.
 */
- (BOOL)setLogSeverityLevel:(ORTLoggingLevel)loggingLevel
                      error:(NSError**)error;

/**
 * Sets a session configuration key-value pair.
 * Any value for a previously set key will be overwritten.
 * The session configuration keys and values are documented here:
 * https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h
 *
 * @param key The key.
 * @param value The value.
 * @param error Optional error information set if an error occurs.
 * @return Whether the option was set successfully.
 */
- (BOOL)addConfigEntryWithKey:(NSString*)key
                        value:(NSString*)value
                        error:(NSError**)error;

/**
 * Registers a block that supplies application-managed external EPContext data during session initialization.
 *
 * Available since 1.30.
 *
 * The block is copied and retained by these options. Each `ORTSession` created from the options takes its own strong
 * registration snapshot for the session lifetime, so replacing or clearing the block only affects future sessions.
 * ONNX Runtime may invoke the block concurrently from arbitrary threads; the block and any captured state must be
 * thread-safe. The returned bytes are copied before the block's returned `NSData` is released.
 *
 * Return zero-length `NSData` for a successful empty payload. Return nil only on failure and set the provided error.
 * Returning nil without an error produces an ONNX Runtime failure status. Data larger than `maxDataSize` is rejected
 * before the native output buffer is allocated.
 *
 * The Objective-C API exposes only the application-side read callback. It does not currently expose model compilation,
 * so the EPContext data write callback is not applicable.
 *
 * @param block The synchronous block used to read EPContext data.
 * @param maxDataSize A finite maximum payload size in bytes. Must be greater than zero and less than `NSUIntegerMax`.
 * @param error Optional error information set if registration fails.
 * @return Whether the block was registered successfully.
 */
- (BOOL)setEpContextDataReadBlock:(ORTEpContextDataReadBlock)block
                      maxDataSize:(NSUInteger)maxDataSize
                            error:(NSError**)error;

/**
 * Clears the block that supplies application-managed external EPContext data for future sessions.
 *
 * Available since 1.30.
 *
 * Sessions already created from these options retain their registration snapshot for the session lifetime.
 *
 * @param error Optional error information set if clearing fails.
 * @return Whether the block was cleared successfully.
 */
- (BOOL)clearEpContextDataReadBlockWithError:(NSError**)error;

/**
 * Registers custom ops for use with `ORTSession`s using this SessionOptions by calling the specified
 * native function name. The custom ops library must either be linked against, or have previously been loaded
 * by the user.
 *
 * Available since 1.14.
 *
 * The registration function must have the signature:
 *    `OrtStatus* (*fn)(OrtSessionOptions* options, const OrtApiBase* api);`
 *
 * The signature is defined in the ONNX Runtime C API:
 * https://github.com/microsoft/onnxruntime/blob/67f4cd54fab321d83e4a75a40efeee95a6a17079/include/onnxruntime/core/session/onnxruntime_c_api.h#L697
 *
 * See https://onnxruntime.ai/docs/reference/operators/add-custom-op.html for more information on custom ops.
 * See https://github.com/microsoft/onnxruntime/blob/342a5bf2b756d1a1fc6fdc582cfeac15182632fe/onnxruntime/test/testdata/custom_op_library/custom_op_library.cc#L115
 * for an example of a custom op library registration function.
 *
 * @note The caller must ensure that `registrationFuncName` names a valid function that is visible to the native ONNX
 * Runtime code and has the correct signature.
 * They must ensure that the function does what they expect it to do because this method will just call it.
 *
 * @param registrationFuncName The name of the registration function to call.
 * @param error Optional error information set if an error occurs.
 * @return Whether the registration function was successfully called.
 */
- (BOOL)registerCustomOpsUsingFunction:(NSString*)registrationFuncName
                                 error:(NSError**)error;

/**
 * Registers custom ops for use with `ORTSession`s using this SessionOptions by calling the specified function
 * pointed to by `registerCustomOpsFn`.
 *
 * Available since 1.16.
 *
 * The registration function must have the signature:
 *    `OrtStatus* (*fn)(OrtSessionOptions* options, const OrtApiBase* api);`
 *
 * The signature is defined in the ONNX Runtime C API:
 * https://github.com/microsoft/onnxruntime/blob/67f4cd54fab321d83e4a75a40efeee95a6a17079/include/onnxruntime/core/session/onnxruntime_c_api.h#L697
 *
 * See https://onnxruntime.ai/docs/reference/operators/add-custom-op.html for more information on custom ops.
 * See https://github.com/microsoft/onnxruntime/blob/342a5bf2b756d1a1fc6fdc582cfeac15182632fe/onnxruntime/test/testdata/custom_op_library/custom_op_library.cc#L115
 * for an example of a custom op library registration function.
 *
 * @note The caller must ensure that `registerCustomOpsFn` is a valid function pointer and has the correct signature.
 * They must ensure that the function does what they expect it to do because this method will just call it.
 *
 * @param registerCustomOpsFn A pointer to the registration function to call.
 * @param error Optional error information set if an error occurs.
 * @return Whether the registration function was successfully called.
 */
- (BOOL)registerCustomOpsUsingFunctionPointer:(ORTCAPIRegisterCustomOpsFnPtr)registerCustomOpsFn
                                        error:(NSError**)error;

/**
 * Registers ONNX Runtime Extensions custom ops that have been built in to ONNX Runtime.
 *
 * Available since 1.16.
 *
 * @note ONNX Runtime must have been built with the `--use_extensions` flag for the ONNX Runtime Extensions custom ops
 * to be able to be registered with this method. When using a separate ONNX Runtime Extensions library, use
 * `registerCustomOpsUsingFunctionPointer:error:` instead.
 *
 * @param error Optional error information set if an error occurs.
 * @return Whether the ONNX Runtime Extensions custom ops were successfully registered.
 */
- (BOOL)enableOrtExtensionsCustomOpsWithError:(NSError**)error;

@end

/**
 * Options for configuring a run.
 */
@interface ORTRunOptions : NSObject

- (instancetype)init NS_UNAVAILABLE;

/**
 * Creates run configuration options.
 *
 * @param error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 */
- (nullable instancetype)initWithError:(NSError**)error NS_SWIFT_NAME(init());

/**
 * Sets the run log tag.
 *
 * @param logTag The log tag.
 * @param error Optional error information set if an error occurs.
 * @return Whether the option was set successfully.
 */
- (BOOL)setLogTag:(NSString*)logTag
            error:(NSError**)error;

/**
 * Sets the run log severity level.
 *
 * @param loggingLevel The log severity level.
 * @param error Optional error information set if an error occurs.
 * @return Whether the option was set successfully.
 */
- (BOOL)setLogSeverityLevel:(ORTLoggingLevel)loggingLevel
                      error:(NSError**)error;

/**
 * Sets a run configuration key-value pair.
 * Any value for a previously set key will be overwritten.
 * The run configuration keys and values are documented here:
 * https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h
 *
 * @param key The key.
 * @param value The value.
 * @param error Optional error information set if an error occurs.
 * @return Whether the option was set successfully.
 */
- (BOOL)addConfigEntryWithKey:(NSString*)key
                        value:(NSString*)value
                        error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
