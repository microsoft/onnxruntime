import {Backend, InferenceSession, SessionHandler} from 'onnxruntime-common';
import {init, OnnxruntimeWebAssemblySessionHandler} from './wasm';

class OnnxruntimeWebAssemblyBackend implements Backend {
  async init(): Promise<void> {
    await init();
  }
  createSessionHandler(path: string, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  createSessionHandler(buffer: Uint8Array, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  async createSessionHandler(pathOrBuffer: string|Uint8Array, _options?: InferenceSession.SessionOptions):
      Promise<SessionHandler> {
    let buffer: Uint8Array;
    if (typeof pathOrBuffer === 'string') {
      const response = await fetch(pathOrBuffer);
      const arrayBuffer = await response.arrayBuffer();
      buffer = new Uint8Array(arrayBuffer);
    } else {
      buffer = pathOrBuffer;
    }
    const handler = new OnnxruntimeWebAssemblySessionHandler();
    // TODO: support SessionOptions
    handler.loadModel(buffer);
    return Promise.resolve(handler);
  }
}

export const wasmBackend = new OnnxruntimeWebAssemblyBackend();
