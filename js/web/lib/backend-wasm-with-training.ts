// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { Backend, env, InferenceSession, TrainingSessionHandler } from 'onnxruntime-common';
import { cpus } from 'os';

import { OnnxruntimeWebAssemblyTrainingSessionHandler } from './wasm/training-session-handler';
import { OnnxruntimeWebAssemblyBackend } from './backend-wasm'

class OnnxruntimeTrainingWebAssemblyBackend extends OnnxruntimeWebAssemblyBackend implements Backend {

  createTrainingSessionHandler(checkpointStateUri: string, trainModelUri: string, evalModelUri?: string,
    optimizerModelUri?: string, options?: InferenceSession.SessionOptions): Promise<TrainingSessionHandler>;
  createTrainingSessionHandler(checkpointStateBuffer: Uint8Array, trainModelBuffer: Uint8Array,
    evalModelBuffer?: Uint8Array, optimizerModelBuffer?: Uint8Array,
    options?: InferenceSession.SessionOptions): Promise<TrainingSessionHandler>;
  async createTrainingSessionHandler(checkpointStateBuffer: string | Uint8Array, trainModelUri: string | Uint8Array,
    evalModelUri?: string | Uint8Array, optimizerModelUri?: string | Uint8Array,
    options?: InferenceSession.SessionOptions): Promise<TrainingSessionHandler> {
    "ImplementThis"
  }
}

export const wasmBackend = new OnnxruntimeTrainingWebAssemblyBackend();
