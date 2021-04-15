import {Backend, InferenceSession, SessionHandler} from 'onnxruntime-common';

class OnnxjsBackend implements Backend {
  async init(): Promise<void> {
    throw new Error('Method not implemented.');
  }
  async createSessionHandler(_pathOrBuffer: string|Uint8Array, _options?: InferenceSession.SessionOptions):
      Promise<SessionHandler> {
    throw new Error('Method not implemented.');
  }
}

export const onnxjsBackend = new OnnxjsBackend();
