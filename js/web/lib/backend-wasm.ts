// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Backend, InferenceSession, TrainingBackend, env, SessionHandler, TrainingSessionHandler, CheckpointHandler} from 'onnxruntime-common';
import {cpus} from 'os';

import {initializeWebAssemblyInstance} from './wasm/proxy-wrapper';
import {OnnxruntimeWebAssemblySessionHandler} from './wasm/session-handler';
import {OnnxruntimeWebAssemblyCheckpointHandler} from './wasm/training-session-handler';

/**
 * This function initializes all flags for WebAssembly.
 *
 * Those flags are accessible from `ort.env.wasm`. Users are allow to set those flags before the first inference session
 * being created, to override default value.
 */
export const initializeFlags = (): void => {
  if (typeof env.wasm.initTimeout !== 'number' || env.wasm.initTimeout < 0) {
    env.wasm.initTimeout = 0;
  }

  if (typeof env.wasm.simd !== 'boolean') {
    env.wasm.simd = true;
  }

  if (typeof env.wasm.proxy !== 'boolean') {
    env.wasm.proxy = false;
  }

  if (typeof env.wasm.numThreads !== 'number' || !Number.isInteger(env.wasm.numThreads) || env.wasm.numThreads <= 0) {
    const numCpuLogicalCores = typeof navigator === 'undefined' ? cpus().length : navigator.hardwareConcurrency;
    env.wasm.numThreads = Math.min(4, Math.ceil((numCpuLogicalCores || 1) / 2));
  }
};

class OnnxruntimeWebAssemblyBackend implements Backend {
  async init(): Promise<void> {
    // populate wasm flags
    initializeFlags();

    // init wasm
    await initializeWebAssemblyInstance();
  }
  createSessionHandler(path: string, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  createSessionHandler(buffer: Uint8Array, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  async createSessionHandler(pathOrBuffer: string|Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<SessionHandler> {
    const handler = new OnnxruntimeWebAssemblySessionHandler();
    await handler.loadModel(pathOrBuffer, options);
    return Promise.resolve(handler);
  }

  // async createTrainingSession(checkpointState: CheckpointState, trainModel: ArrayBufferLike|string, evalModel: ArrayBufferLike|string,
  //     optimizerModel: ArrayBufferLike|string, options?: Session.SessionOptions): Promise<TrainingSessionHandler> {
  //   const handler = new OnnxruntimeWebAssemblyTrainingSessionHandler();
  //   await handler.loadTrainingSession(checkpointState, trainModel, evalModel, optimizerModel, options);
  //   return Promise.resolve(handler);
  // }

}

class OnnxruntimeWebAssemblyTrainingBackend extends OnnxruntimeWebAssemblyBackend implements TrainingBackend {
  createSessionHandler(path: string, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  createSessionHandler(buffer: Uint8Array, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  async createSessionHandler(pathOrBuffer: string|Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<SessionHandler> {
        throw new Error('Can\'t use the training backend to create an inference session');
      }

  async createCheckpointHandler(pathOrBuffer: string|Uint8Array): Promise<CheckpointHandler> {
    const handler = new OnnxruntimeWebAssemblyCheckpointHandler();
    await handler.loadCheckpoint(pathOrBuffer);
    return Promise.resolve(handler);
  }
}

export const wasmBackend = new OnnxruntimeWebAssemblyBackend();
