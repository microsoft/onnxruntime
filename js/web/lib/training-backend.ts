// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import {TrainingBackend, InferenceSession, SessionHandler, TrainingHandler} from 'onnxruntime-common';
import {OnnxruntimeWebAssemblyBackend as WebAssemblyInferenceBackend} from './backend-wasm';
import { OnnxruntimeWebAssemblyTrainingHandler } from './wasm/training-session-handler';

class WebAssemblyTrainingBackend extends WebAssemblyInferenceBackend implements TrainingBackend {

  createSessionHandler(path: string, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  createSessionHandler(buffer: Uint8Array, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  async createSessionHandler(pathOrBuffer: string|Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<SessionHandler> {
        throw new Error('Can\'t use the training backend to create an inference session');
      }

  async loadCheckpoint(pathOrBuffer: string|Uint8Array): Promise<TrainingHandler> {
    console.log("inside the web training backend");
    const handler = new OnnxruntimeWebAssemblyTrainingHandler();
    await handler.loadCheckpoint(pathOrBuffer);
    return Promise.resolve(handler);
  }

  // async createTrainingSession(checkpointState: CheckpointState, trainModel: ArrayBufferLike|string, evalModel: ArrayBufferLike|string,
  //     optimizerModel: ArrayBufferLike|string, options?: Session.SessionOptions): Promise<TrainingSessionHandler> {
  //   const handler = new OnnxruntimeWebAssemblyTrainingSessionHandler();
  //   await handler.loadTrainingSession(checkpointState, trainModel, evalModel, optimizerModel, options);
  //   return Promise.resolve(handler);
  // }
}

export const trainingWasmBackend = new WebAssemblyTrainingBackend();
