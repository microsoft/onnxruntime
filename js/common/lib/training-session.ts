// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession} from './inference-session.js';
import {OnnxValue} from './onnx-value.js';
import {TrainingSession as TrainingSessionImpl} from './training-session-impl.js';

/* eslint-disable @typescript-eslint/no-redeclare */

export declare namespace TrainingSession {
  /**
   * Either URI file path (string) or Uint8Array containing model or checkpoint information.
   */
  type URIorBuffer = string|Uint8Array;
}

/**
 * Represent a runtime instance of an ONNX training session,
 * which contains a model that can be trained, and, optionally,
 * an eval and optimizer model.
 */
export interface TrainingSession {
  // #region run()

  /**
   * Lazily resets the gradients of all trainable parameters to zero. Should happen before the invocation of the next
   * runTrainStep().
   */
  lazyResetGrad(): Promise<void>;

  /**
   * Run TrainStep asynchronously with the given feeds and options.
   *
   * @param feeds - Representation of the model input. See type description of `InferenceSession.InputType` for
   detail.
   * @param options - Optional. A set of options that controls the behavior of model training.
   * @returns A promise that resolves to a map, which uses output names as keys and OnnxValue as corresponding values.
   */
  runTrainStep(feeds: InferenceSession.FeedsType, options?: InferenceSession.RunOptions):
      Promise<InferenceSession.ReturnType>;

  /**
   * Run a single train step with the given inputs and options.
   *
   * @param feeds - Representation of the model input.
   * @param fetches - Representation of the model output.
   * detail.
   * @param options - Optional. A set of options that controls the behavior of model training.
   * @returns A promise that resolves to a map, which uses output names as keys and OnnxValue as corresponding
   values.
   */
  runTrainStep(
      feeds: InferenceSession.FeedsType, fetches: InferenceSession.FetchesType,
      options?: InferenceSession.RunOptions): Promise<InferenceSession.ReturnType>;

  /**
   * Runs a single optimizer step, which performs weight updates for the trainable parameters using the optimizer model.
   *
   * @param options - Optional. A set of options that controls the behavior of model optimizing.
   */
  runOptimizerStep(options?: InferenceSession.RunOptions): Promise<void>;

  /**
   * Run a single eval step with the given inputs and options using the eval model.
   *
   * @param feeds - Representation of the model input.
   * @param options - Optional. A set of options that controls the behavior of model eval step.
   * @returns A promise that resolves to a map, which uses output names as keys and OnnxValue as corresponding
   values.
   */
  runEvalStep(feeds: InferenceSession.FeedsType, options?: InferenceSession.RunOptions):
      Promise<InferenceSession.ReturnType>;

  /**
   * Run a single eval step with the given inputs and options using the eval model.
   *
   * @param feeds - Representation of the model input.
   * @param fetches - Representation of the model output.
   * detail.
   * @param options - Optional. A set of options that controls the behavior of model eval step.
   * @returns A promise that resolves to a map, which uses output names as keys and OnnxValue as corresponding
   values.
   */
  runEvalStep(
      feeds: InferenceSession.FeedsType, fetches: InferenceSession.FetchesType,
      options?: InferenceSession.RunOptions): Promise<InferenceSession.ReturnType>;

  // #endregion

  // #region copy parameters

  /**
   * Retrieves the size of all parameters for the training state. Calculates the total number of primitive (datatype of
   * the parameters) elements of all the parameters in the training state.
   *
   * @param trainableOnly - When set to true, the size is calculated for trainable params only. Default value is true.
   */
  getParametersSize(trainableOnly: boolean): Promise<number>;

  /**
   * Copies parameter values from the given array to the training state. Currently, only supporting models with
   * parameters of type Float32.
   *
   * @param buffer - Float32 buffer containing parameters converted to a Uint8Array.
   * @param trainableOnly - True if trainable parameters only to be modified, false otherwise. Default value is true.
   */
  loadParametersBuffer(array: Uint8Array, trainableOnly: boolean): Promise<void>;

  /**
   * Copies the model parameters to a contiguous buffer. Usually used in the context of Federated Learning.
   * Currently, only supporting models with parameters of type Float32.
   *
   * @param trainableOnly - When set to true, only trainable parameters are copied. Trainable parameters are parameters
   * for which requires_grad is set to true. Default value is true.
   * @returns A promise that resolves to a Float32 OnnxValue of the requested parameters.
   */
  getContiguousParameters(trainableOnly: boolean): Promise<OnnxValue>;
  // #endregion

  // #region release()

  /**
   * Release the inference session and the underlying resources.
   */
  release(): Promise<void>;
  // #endregion

  // #region metadata

  /**
   * Get input names of the loaded training model.
   */
  readonly trainingInputNames: readonly string[];

  /**
   * Get output names of the loaded training model.
   */
  readonly trainingOutputNames: readonly string[];

  /**
   * Get input names of the loaded eval model. Is an empty array if no eval model is loaded.
   */
  readonly evalInputNames: readonly string[];

  /**
   * Get output names of the loaded eval model. Is an empty array if no eval model is loaded.
   */
  readonly evalOutputNames: readonly string[];

  // #endregion
}

/**
 * Represents the optional parameters that can be passed into the TrainingSessionFactory.
 */
export interface TrainingSessionCreateOptions {
  /**
   * URI or buffer for a .ckpt file that contains the checkpoint for the training model.
   */
  checkpointState: TrainingSession.URIorBuffer;
  /**
   * URI or buffer for the .onnx training file.
   */
  trainModel: TrainingSession.URIorBuffer;
  /**
   * Optional. URI or buffer for the .onnx optimizer model file.
   */
  optimizerModel?: TrainingSession.URIorBuffer;
  /**
   * Optional. URI or buffer for the .onnx eval model file.
   */
  evalModel?: TrainingSession.URIorBuffer;
}

/**
 * Defines method overload possibilities for creating a TrainingSession.
 */
export interface TrainingSessionFactory {
  // #region create()

  /**
   * Creates a new TrainingSession and asynchronously loads any models passed in through trainingOptions
   *
   * @param trainingOptions specify models and checkpoints to load into the Training Session
   * @param sessionOptions specify configuration for training session behavior
   *
   * @returns Promise that resolves to a TrainingSession object
   */
  create(trainingOptions: TrainingSessionCreateOptions, sessionOptions?: InferenceSession.SessionOptions):
      Promise<TrainingSession>;

  // #endregion
}

// eslint-disable-next-line @typescript-eslint/naming-convention
export const TrainingSession: TrainingSessionFactory = TrainingSessionImpl;
