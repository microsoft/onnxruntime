// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>
#include <stdint.h>

NS_ASSUME_NONNULL_BEGIN

@class ORTCheckpoint;
@class ORTEnv;
@class ORTValue;
@class ORTSessionOptions;

/**
 * Trainer class that provides methods to train, evaluate and optimize ONNX models.
 *
 * The training session requires four training artifacts:
 *  1. Training onnx model
 *  2. Evaluation onnx model (optional)
 *  3. Optimizer onnx model
 *  4. Checkpoint directory
 *
 * [onnxruntime-training python utility](https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/python/training/onnxblock/README.md)
 * can be used to generate above training artifacts.
 *
 * Available since 1.16.
 *
 * @note This class is only available when the training APIs are enabled.
 */
@interface ORTTrainingSession : NSObject

- (instancetype)init NS_UNAVAILABLE;

/**
 * Creates a training session from the training artifacts that can be used to begin or resume training.
 *
 * The initializer instantiates the training session based on provided env and session options, which can be used to
 * begin or resume training from a given checkpoint state. The checkpoint state represents the parameters of training
 * session which will be moved to the device specified in the session option if needed.
 *
 * @param env The `ORTEnv` instance to use for the training session.
 * @param sessionOptions The `ORTSessionOptions` to use for the training session.
 * @param checkpoint Training states that are used as a starting point for training.
 * @param trainModelPath The path to the training onnx model.
 * @param evalModelPath The path to the evaluation onnx model.
 * @param optimizerModelPath The path to the optimizer onnx model used to perform gradient descent.
 * @param error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 *
 * @note Note that the training session created with a checkpoint state uses this state to store the entire training
 * state (including model parameters, its gradients, the optimizer states and the properties). The training session
 * keeps a strong (owning) pointer to the checkpoint state.
 */
- (nullable instancetype)initWithEnv:(ORTEnv*)env
                      sessionOptions:(ORTSessionOptions*)sessionOptions
                          checkpoint:(ORTCheckpoint*)checkpoint
                      trainModelPath:(NSString*)trainModelPath
                       evalModelPath:(nullable NSString*)evalModelPath
                  optimizerModelPath:(nullable NSString*)optimizerModelPath
                               error:(NSError**)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs a training step, which is equivalent to a forward and backward propagation in a single step.
 *
 * The training step computes the outputs of the training model and the gradients of the trainable parameters
 * for the given input values. The train step is performed based on the training model that was provided to the training session.
 * It is equivalent to running forward and backward propagation in a single step. The computed gradients are stored inside
 * the training session state so they can be later consumed by `optimizerStep`. The gradients can be lazily reset by
 * calling `lazyResetGrad` method.
 *
 * @param inputs The input values to the training model.
 * @param error Optional error information set if an error occurs.
 * @return The output values of the training model.
 */
- (nullable NSArray<ORTValue*>*)trainStepWithInputValues:(NSArray<ORTValue*>*)inputs
                                                   error:(NSError**)error;

/**
 * Performs a evaluation step that computes the outputs of the evaluation model for the given inputs.
 * The eval step is performed based on the evaluation model that was provided to the training session.
 *
 * @param inputs The input values to the eval model.
 * @param error Optional error information set if an error occurs.
 * @return The output values of the eval model.
 *
 */
- (nullable NSArray<ORTValue*>*)evalStepWithInputValues:(NSArray<ORTValue*>*)inputs
                                                  error:(NSError**)error;

/**
 * Reset the gradients of all trainable parameters to zero lazily.
 *
 * Calling this method sets the internal state of the training session such that the gradients of the trainable parameters
 * in the ORTCheckpoint will be scheduled to be reset just before the new gradients are computed on the next
 * invocation of the `trainStep` method.
 *
 * @param error Optional error information set if an error occurs.
 * @return YES if the gradients are set to reset successfully, NO otherwise.
 */
- (BOOL)lazyResetGradWithError:(NSError**)error;

/**
 * Performs the weight updates for the trainable parameters using the optimizer model. The optimizer step is performed
 * based on the optimizer model that was provided to the training session. The updated parameters are stored inside the
 * training state so that they can be used by the next `trainStep` method call.
 *
 * @param error Optional error information set if an error occurs.
 * @return YES if the optimizer step was performed successfully, NO otherwise.
 */
- (BOOL)optimizerStepWithError:(NSError**)error;

/**
 * Returns the names of the user inputs for the training model that can be associated with
 * the `ORTValue` provided to the `trainStep`.
 *
 * @param error Optional error information set if an error occurs.
 * @return The names of the user inputs for the training model.
 */
- (nullable NSArray<NSString*>*)getTrainInputNamesWithError:(NSError**)error;

/**
 * Returns the names of the user inputs for the evaluation model that can be associated with
 * the `ORTValue` provided to the `evalStep`.
 *
 * @param error Optional error information set if an error occurs.
 * @return The names of the user inputs for the evaluation model.
 */
- (nullable NSArray<NSString*>*)getEvalInputNamesWithError:(NSError**)error;

/**
 * Returns the names of the user outputs for the training model that can be associated with
 * the `ORTValue` returned by the `trainStep`.
 *
 * @param error Optional error information set if an error occurs.
 * @return The names of the user outputs for the training model.
 */
- (nullable NSArray<NSString*>*)getTrainOutputNamesWithError:(NSError**)error;

/**
 * Returns the names of the user outputs for the evaluation model that can be associated with
 * the `ORTValue` returned by the `evalStep`.
 *
 * @param error Optional error information set if an error occurs.
 * @return The names of the user outputs for the evaluation model.
 */
- (nullable NSArray<NSString*>*)getEvalOutputNamesWithError:(NSError**)error;

/**
 * Registers a linear learning rate scheduler for the training session.
 *
 * The scheduler gradually decreases the learning rate from the initial value to zero over the course of the training.
 * The decrease is performed by multiplying the current learning rate by a linearly updated factor.
 * Before the decrease, the learning rate is gradually increased from zero to the initial value during a warmup phase.
 *
 * @param warmupStepCount The number of steps to perform the linear warmup.
 * @param totalStepCount The total number of steps to perform the linear decay.
 * @param initialLr The initial learning rate.
 * @param error Optional error information set if an error occurs.
 * @return YES if the scheduler was registered successfully, NO otherwise.
 */
- (BOOL)registerLinearLRSchedulerWithWarmupStepCount:(int64_t)warmupStepCount
                                      totalStepCount:(int64_t)totalStepCount
                                           initialLr:(float)initialLr
                                               error:(NSError**)error;

/**
 * Update the learning rate based on the registered learning rate scheduler.
 *
 * Performs a scheduler step that updates the learning rate that is being used by the training session.
 * This function should typically be called before invoking the optimizer step for each round, or as necessary
 * to update the learning rate being used by the training session.
 *
 * @note A valid predefined learning rate scheduler must be first registered to invoke this method.
 *
 * @param error Optional error information set if an error occurs.
 * @return YES if the scheduler step was performed successfully, NO otherwise.
 */
- (BOOL)schedulerStepWithError:(NSError**)error;

/**
 * Returns the current learning rate being used by the training session.
 *
 * @param error Optional error information set if an error occurs.
 * @return The current learning rate or 0.0f if an error occurs.
 */
- (float)getLearningRateWithError:(NSError**)error __attribute__((swift_error(nonnull_error)));

/**
 * Sets the learning rate being used by the training session.
 *
 * The current learning rate is maintained by the training session and can be overwritten by invoking this method
 * with the desired learning rate. This function should not be used when a valid learning rate scheduler is registered.
 * It should be used either to set the learning rate derived from a custom learning rate scheduler or to set a constant
 * learning rate to be used throughout the training session.
 *
 * @note It does not set the initial learning rate that may be needed by the predefined learning rate schedulers.
 * To set the initial learning rate for learning rate schedulers, use the `registerLinearLRScheduler` method.
 *
 * @param lr The learning rate to be used by the training session.
 * @param error Optional error information set if an error occurs.
 * @return YES if the learning rate was set successfully, NO otherwise.
 */
- (BOOL)setLearningRate:(float)lr
                  error:(NSError**)error;

/**
 * Loads the training session model parameters from a contiguous buffer.
 *
 * @param buffer Contiguous buffer to load the parameters from.
 * @param error Optional error information set if an error occurs.
 * @return YES if the parameters were loaded successfully, NO otherwise.
 */
- (BOOL)fromBufferWithValue:(ORTValue*)buffer
                      error:(NSError**)error;

/**
 * Returns a contiguous buffer that holds a copy of all training state parameters.
 *
 * @param onlyTrainable If YES, returns a buffer that holds only the trainable parameters, otherwise returns a buffer
 * that holds all the parameters.
 * @param error Optional error information set if an error occurs.
 * @return A contiguous buffer that holds a copy of all training state parameters.
 */
- (nullable ORTValue*)toBufferWithTrainable:(BOOL)onlyTrainable
                                      error:(NSError**)error;

/**
 * Exports the training session model that can be used for inference.
 *
 * If the training session was provided with an eval model, the training session can generate an inference model if it
 * knows the inference graph outputs. The input inference graph outputs are used to prune the eval model so that the
 * inference model's outputs align with the provided outputs. The exported model is saved at the path provided and
 * can be used for inferencing with `ORTSession`.
 *
 * @note The method reloads the eval model from the path provided to the initializer and expects this path to be valid.
 *
 * @param inferenceModelPath The path to the serialized the inference model.
 * @param graphOutputNames The names of the outputs that are needed in the inference model.
 * @param error Optional error information set if an error occurs.
 * @return YES if the inference model was exported successfully, NO otherwise.
 */
- (BOOL)exportModelForInferenceWithOutputPath:(NSString*)inferenceModelPath
                             graphOutputNames:(NSArray<NSString*>*)graphOutputNames
                                        error:(NSError**)error;
@end

#ifdef __cplusplus
extern "C" {
#endif

/**
 * This function sets the seed for generating random numbers.
 * Use this function to generate reproducible results. It should be noted that completely reproducible results are not guaranteed.
 *
 * @param seed Manually set seed to use for random number generation.
 */
void ORTSetSeed(int64_t seed);

#ifdef __cplusplus
}
#endif

NS_ASSUME_NONNULL_END
