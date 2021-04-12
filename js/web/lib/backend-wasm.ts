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
    if (typeof pathOrBuffer === 'string') {
      const response = await fetch(pathOrBuffer);
      const arrayBuffer = await response.arrayBuffer();
      pathOrBuffer = new Uint8Array(arrayBuffer);
    }
    const handler = new OnnxruntimeWebAssemblySessionHandler();
    // TODO: support SessionOptions
    handler.loadModel(pathOrBuffer);
    return Promise.resolve(handler);
  }
}

export const wasmBackend = new OnnxruntimeWebAssemblyBackend();
