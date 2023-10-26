// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession, TrainingSessionHandler} from 'onnxruntime-common';

import {OnnxruntimeWebAssemblyBackend} from './backend-wasm';
import {OnnxruntimeWebAssemblyTrainingSessionHandler} from './wasm/session-handler-for-training';

class OnnxruntimeTrainingWebAssemblyBackend extends OnnxruntimeWebAssemblyBackend {
  async createTrainingSessionHandler(
      checkpointStateUriOrBuffer: string|Uint8Array, trainModelUriOrBuffer: string|Uint8Array,
      evalModelUriOrBuffer: string|Uint8Array, optimizerModelUriOrBuffer: string|Uint8Array,
      options: InferenceSession.SessionOptions): Promise<TrainingSessionHandler> {
    const handler = new OnnxruntimeWebAssemblyTrainingSessionHandler();
    await handler.createTrainingSession(
        checkpointStateUriOrBuffer, trainModelUriOrBuffer, evalModelUriOrBuffer, optimizerModelUriOrBuffer, options);
    return Promise.resolve(handler);
  }
}

export const wasmBackend = new OnnxruntimeTrainingWebAssemblyBackend();
