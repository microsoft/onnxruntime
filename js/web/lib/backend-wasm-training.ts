// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession, TrainingSessionHandler} from 'onnxruntime-common';

import {OnnxruntimeWebAssemblyBackend} from './backend-wasm';

class OnnxruntimeTrainingWebAssemblyBackend extends OnnxruntimeWebAssemblyBackend {
  async createTrainingSessionHandler(
      _checkpointStateUriOrBuffer: string|Uint8Array, _trainModelUriOrBuffer: string|Uint8Array,
      _evalModelUriOrBuffer: string|Uint8Array, _optimizerModelUriOrBuffer: string|Uint8Array,
      _options: InferenceSession.SessionOptions): Promise<TrainingSessionHandler> {
    throw new Error('Method not implemented yet.');
  }
}

export const wasmBackend = new OnnxruntimeTrainingWebAssemblyBackend();
