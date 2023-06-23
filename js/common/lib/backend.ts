// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Session} from './inference-session';
import {OnnxValue} from './onnx-value';
import {CheckpointState} from './training-session';

/**
 * @internal
 */
export declare namespace SessionHandler {
  type FeedsType = {[name: string]: OnnxValue};
  type FetchesType = {[name: string]: OnnxValue | null};
  type ReturnType = {[name: string]: OnnxValue};
}

/**
 * Represent a handler instance of an inference session.
 *
 * @internal
 */
export interface SessionHandler {
  dispose(): Promise<void>;

  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];

  startProfiling(): void;
  endProfiling(): void;

  run(feeds: SessionHandler.FeedsType, fetches: SessionHandler.FetchesType,
      options: Session.RunOptions): Promise<SessionHandler.ReturnType>;
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

  createSessionHandler(uriOrBuffer: string|Uint8Array, options?: Session.SessionOptions):
      Promise<SessionHandler>;

  createCheckpointState(pathOrBuffer: string|Uint8Array): Promise<CheckpointState>;

  createTrainingSession(checkpointState: CheckpointState, trainModel: ArrayBufferLike|string, evalModel: ArrayBufferLike|string,
      optimizerModel: ArrayBufferLike|string, options?: Session.SessionOptions): Promise<TrainingSessionHandler>;
}

export {registerBackend} from './backend-impl';

export interface CheckpointHandler {
  // save checkpoint implementation would go here
  // need class representation of a checkpoint handler to also have the number id for handle
  dispose(): Promise<void>;
}

export interface TrainingSessionHandler {
  dispose(): Promise<void>;

}
