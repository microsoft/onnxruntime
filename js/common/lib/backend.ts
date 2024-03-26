// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession} from './inference-session.js';
import {OnnxValue} from './onnx-value.js';
import {TrainingSession} from './training-session.js';

/**
 * @ignore
 */
export declare namespace SessionHandler {
  type FeedsType = {[name: string]: OnnxValue};
  type FetchesType = {[name: string]: OnnxValue | null};
  type ReturnType = {[name: string]: OnnxValue};
}

/**
 * Represents shared SessionHandler functionality
 *
 * @ignore
 */
interface SessionHandler {
  dispose(): Promise<void>;

  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
}

/**
 * Represent a handler instance of an inference session.
 *
 * @ignore
 */
export interface InferenceSessionHandler extends SessionHandler {
  startProfiling(): void;
  endProfiling(): void;

  run(feeds: SessionHandler.FeedsType, fetches: SessionHandler.FetchesType,
      options: InferenceSession.RunOptions): Promise<SessionHandler.ReturnType>;
}

/**
 * Represent a handler instance of a training inference session.
 *
 * @ignore
 */
export interface TrainingSessionHandler extends SessionHandler {
  readonly evalInputNames: readonly string[];
  readonly evalOutputNames: readonly string[];

  lazyResetGrad(): Promise<void>;
  runTrainStep(
      feeds: SessionHandler.FeedsType, fetches: SessionHandler.FetchesType,
      options: InferenceSession.RunOptions): Promise<SessionHandler.ReturnType>;
  runOptimizerStep(options: InferenceSession.RunOptions): Promise<void>;
  runEvalStep(
      feeds: SessionHandler.FeedsType, fetches: SessionHandler.FetchesType,
      options: InferenceSession.RunOptions): Promise<SessionHandler.ReturnType>;

  getParametersSize(trainableOnly: boolean): Promise<number>;
  loadParametersBuffer(buffer: Uint8Array, trainableOnly: boolean): Promise<void>;
  getContiguousParameters(trainableOnly: boolean): Promise<OnnxValue>;
}

/**
 * Represent a backend that provides implementation of model inferencing.
 *
 * @ignore
 */
export interface Backend {
  /**
   * Initialize the backend asynchronously. Should throw when failed.
   */
  init(backendName: string): Promise<void>;

  createInferenceSessionHandler(uriOrBuffer: string|Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<InferenceSessionHandler>;

  createTrainingSessionHandler?
      (checkpointStateUriOrBuffer: TrainingSession.UriOrBuffer, trainModelUriOrBuffer: TrainingSession.UriOrBuffer,
       evalModelUriOrBuffer: TrainingSession.UriOrBuffer, optimizerModelUriOrBuffer: TrainingSession.UriOrBuffer,
       options: InferenceSession.SessionOptions): Promise<TrainingSessionHandler>;
}

export {registerBackend} from './backend-impl.js';
