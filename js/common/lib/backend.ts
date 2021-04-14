// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession} from './inference-session';
import {OnnxValue} from './onnx-value';

/**
 * Represent a handler instance of an inference session.
 */
export interface SessionHandler {
  dispose(): Promise<void>;

  readonly inputNames: string[];
  readonly outputNames: string[];

  run(feeds: {[name: string]: OnnxValue}, fetches: {[name: string]: OnnxValue|null},
      options: InferenceSession.RunOptions): Promise<{[name: string]: OnnxValue}>;
}

/**
 * Represent a backend that provides implementation of model inferencing.
 */
export interface Backend {
  /**
   * Initialize the backend asynchronously. Should throw when failed.
   */
  init(): Promise<void>;

  createSessionHandler(uriOrBuffer: string|Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<SessionHandler>;
}

export {registerBackend} from './backend-impl';
