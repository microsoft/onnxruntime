// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession} from './inference-session.js';
import {TrainingSession as TrainingSessionImpl} from './training-session-impl.js';

/* eslint-disable @typescript-eslint/no-redeclare */

/**
 * Represent a runtime instance of an ONNX training session,
 * which contains a model that can be trained, and, optionally,
 * an eval and optimizer model.
 */
export interface TrainingSession {
  // #region run()

  // /**
  //  * Run TrainStep asynchronously with the given feeds and options.
  //  *
  //  * @param feeds - Representation of the model input. See type description of `InferenceSession.InputType` for
  //  detail.
  //  * @param options - Optional. A set of options that controls the behavior of model training.
  //  * @returns A promise that resolves to an error code
  //  */
  // runTrainStep(feeds: InferenceSession.FeedsType, options?: InferenceSession.RunOptions):
  // Promise<InferenceSession.ReturnType>;

  // /**
  //  * Run a single train step with the given inputs and options.
  //  *
  //  * @param feeds - Representation of the model input.
  //  * @param fetches - Representation of the model output.
  //  * detail.
  //  * @param options - Optional. A set of options that controls the behavior of model inference.
  //  * @returns A promise that resolves to a map, which uses output names as keys and OnnxValue as corresponding
  //  values.
  //  */
  // runTrainStep(feeds: InferenceSession.FeedsType, fetches: InferenceSession.FetchesType,
  //   options?: InferenceSession.RunOptions): Promise<InferenceSession.ReturnType>;

  // #endregion

  // #region copy parameters
  // /**
  //  * Copies from a buffer containing parameters to the TrainingSession parameters.
  //  * @param buffer - buffer containing parameters
  //  * @param trainableOnly - Optional. True if trainable parameters only to be modified, false otherwise.
  //  */
  // copyParametersFromBuffer(buffer: ArrayBufferLike, trainableOnly: boolean): void;

  // /**
  //  * Copies from the TrainingSession parameters to the given buffer.
  //  * @param buffer - buffer that will be modified. Must be the same size as results of GetParametersSize API call.
  //  * @param trainableOnly - Optional. True if trainable parameters only to be copied, false othrwise.
  //  */
  // copyParametersToBuffer(buffer: ArrayBufferLike, trainableOnly: boolean): void;
  // #endregion

  // #region release()

  /**
   * Release the inference session and the underlying resources.
   */
  release(): Promise<void>;
  // #endregion

  // #region metadata

  /**
   * Get input names of the loaded model.
   */
  readonly inputNames: readonly string[];

  /**
   * Get output names of the loaded model.
   */
  readonly outputNames: readonly string[];
  // #endregion
}

/**
 * Defines method overload possibilities for creating a TrainingSession.
 */
export interface TrainingSessionFactory {
  // #region create()

  /**
   * Create a new training session and load models asynchronously from an ONNX model file.
   *
   * @param uri - The URI or file path of the model to load.
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(checkpointStateUri: string, trainModelURI: string, options?: InferenceSession.SessionOptions):
      Promise<TrainingSession>;

  /**
   * Create a new training session and load models asynchronously from an ONNX model file.
   *
   * @param buffer - an ArrayBuffer representation of the model
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(
      checkpointStateBuffer: ArrayBufferLike, trainModelBuffer: ArrayBufferLike,
      options?: InferenceSession.SessionOptions): Promise<TrainingSession>;

  /**
   * Create a new training session and load models asynchronously from an ONNX model file.
   *
   * @param buffer - a Uint8Array representation of the model
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(checkpointStateBuffer: Uint8Array, trainModelBuffer: Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<TrainingSession>;

  /**
   * Create a new training session and load models asynchronously from an ONNX model file.
   *
   * @param uri - The URI or file path of the model to load.
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(
      checkpointStateUri: string, trainModelURI: string, optimizerModelURI?: string,
      options?: InferenceSession.SessionOptions): Promise<TrainingSession>;

  /**
   * Create a new training session and load models asynchronously from an ONNX model file.
   *
   * @param buffer - an ArrayBufferLike representation of the model
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(
      checkpointStateBuffer: ArrayBufferLike, trainModelBuffer: ArrayBufferLike, optimizerModelBuffer?: ArrayBufferLike,
      options?: InferenceSession.SessionOptions): Promise<TrainingSession>;

  /**
   * Create a new training session and load models asynchronously from an ONNX model file.
   *
   * @param buffer - a Uint8Array representation of the model
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(
      checkpointStateBuffer: Uint8Array, trainModelBuffer: Uint8Array, optimizerModelBuffer?: Uint8Array,
      options?: InferenceSession.SessionOptions): Promise<TrainingSession>;

  /**
   * Create a new training session and load models asynchronously from an ONNX model file.
   *
   * @param uri - The URI or file path of the model to load.
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(
      checkpointStateUri: string, trainModelURI: string, optimizerModelURI?: string, evalModelURI?: string,
      options?: InferenceSession.SessionOptions): Promise<TrainingSession>;

  /**
   * Create a new training session and load model asynchronously from an array bufer.
   *
   * @param buffer - An ArrayBuffer representation of an ONNX model.
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(
      checkpointStateBuffer: ArrayBufferLike, trainModelBuffer: ArrayBufferLike, optimizerModelBuffer?: ArrayBufferLike,
      evalModelBuffer?: ArrayBufferLike, options?: InferenceSession.SessionOptions): Promise<TrainingSession>;

  /**
   * Create a new training session and load model asynchronously from a Uint8Array.
   *
   * @param buffer - A Uint8Array representation of an ONNX model.
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(
      checkpointState: Uint8Array, trainModelData: Uint8Array, optimizerModelData?: Uint8Array,
      evalModelData?: Uint8Array, options?: InferenceSession.SessionOptions): Promise<TrainingSession>;

  // #endregion
}

// eslint-disable-next-line @typescript-eslint/naming-convention
export const TrainingSession: TrainingSessionFactory = TrainingSessionImpl;
