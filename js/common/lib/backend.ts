// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession} from './inference-session.js';
import {OnnxValue} from './onnx-value.js';

/**
 * @internal
 */
export declare namespace SessionHandler {
  type FeedsType = {[name: string]: OnnxValue};
  type FetchesType = {[name: string]: OnnxValue | null};
  type ReturnType = {[name: string]: OnnxValue};
}

/**
 * Represents shared SessionHandler functionality
 *
 * @internal
 */
interface SessionHandler {
  dispose(): Promise<void>;

  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
}

/**
 * Represent a handler instance of an inference session.
 *
 * @internal
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
 * @internal
 */
export interface TrainingSessionHandler extends SessionHandler {
  dispose(): Promise<void>;

  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];

  runTrainStep(
      feeds: SessionHandler.FeedsType, fetches: SessionHandler.FetchesType,
      options: InferenceSession.RunOptions): Promise<SessionHandler.ReturnType>;

  setParameters(buffer: ArrayBufferLike, trainableOnly: boolean): void;
  getParameters(trainableOnly: boolean): Promise<ArrayBufferLike>;
}

/**
 * Represent a backend that provides implementation of model inferencing.
 *
 * @internal
 */
export interface Backend {
  /**
   * Initialize the backend asynchronously. Should throw when failed.
   */
  init(): Promise<void>;

  createInferenceSessionHandler(uriOrBuffer: string|Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<InferenceSessionHandler>;

  createTrainingSessionHandler?
      (checkpointStateUriOrBuffer: string|Uint8Array, trainModelUriOrBuffer: string|Uint8Array,
       evalModelUriOrBuffer: string|Uint8Array, optimizerModelUriOrBuffer: string|Uint8Array,
       options: InferenceSession.SessionOptions): Promise<TrainingSessionHandler>;
}

export {registerBackend} from './backend-impl.js';
