// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession} from './inference-session';
import {OnnxValue} from './onnx-value';

export interface SessionHandler {
  dispose(): Promise<void>;

  readonly inputNames: string[];
  readonly outputNames: string[];

  run(feeds: SessionHandler.FeedsType, fetches: SessionHandler.FetchesType,
      options: InferenceSession.RunOptions): Promise<SessionHandler.ReturnType>;
}

export declare namespace SessionHandler {
  type FeedsType = {[name: string]: OnnxValue};
  type FetchesType = {[name: string]: OnnxValue | null};
  type ReturnType = {[name: string]: OnnxValue};
}

export interface Backend {
  init(): Promise<void>;

  createSessionHandler(path: string, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  createSessionHandler(buffer: Uint8Array, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
}

const backends: {[name: string]: Backend} = {};

export const registerBackend = (name: string, backend: Backend): void => {
  if (backend && typeof backend.init === 'function' && typeof backend.createSessionHandler === 'function') {
    backends[name] = backend;
  }
};

export const resolveBackend = async(options?: InferenceSession.SessionOptions): Promise<Backend> => {
  if (!options || !options.executionProviders || options.executionProviders.length === 0) {
    await backends.cpu.init();
    return backends.cpu;
  }

  for (let ep of options.executionProviders) {
    if (typeof ep !== 'string' && ep.name) {
      ep = ep.name;
    }
    const backend = backends[ep];
    if (backend) {
      await backend.init();
      return backend;
    }
  }

  throw new Error('no available backend found.');
};
