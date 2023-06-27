// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TrainingBackend, OnnxruntimeWebAssemblyBackend, CheckpointHandler, InferenceSession,
    SessionHandler, OnnxruntimeWebAssemblyCheckpointHandler} from 'onnxruntime-common';

class WebAssemblyTrainingBackend extends OnnxruntimeWebAssemblyBackend implements TrainingBackend {
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
