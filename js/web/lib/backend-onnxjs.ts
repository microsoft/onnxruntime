import {Backend, InferenceSession, SessionHandler} from 'onnxruntime-common';

class OnnxjsBackend implements Backend {
  init(): Promise<void> {
    throw new Error('Method not implemented.');
  }
  createSessionHandler(path: string, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  createSessionHandler(buffer: Uint8Array, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  createSessionHandler(_pathOrBuffer: string|Uint8Array, _options?: InferenceSession.SessionOptions):
      Promise<SessionHandler> {
    throw new Error('Method not implemented.');
  }
}

export const onnxjsBackend = new OnnxjsBackend();
