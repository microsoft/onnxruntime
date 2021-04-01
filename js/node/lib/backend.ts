import {Backend, InferenceSession, SessionHandler} from 'onnxruntime-common';
import {Binding, binding} from './binding';

class OnnxruntimeSessionHandler implements SessionHandler {
  #inferenceSession: Binding.InferenceSession;

  constructor(pathOrBuffer: string|Uint8Array, options: InferenceSession.SessionOptions) {
    this.#inferenceSession = new binding.InferenceSession();
    if (typeof pathOrBuffer === 'string') {
      this.#inferenceSession.loadModel(pathOrBuffer, options);
    } else {
      this.#inferenceSession.loadModel(pathOrBuffer.buffer, pathOrBuffer.byteOffset, pathOrBuffer.byteLength, options);
    }
    this.inputNames = this.#inferenceSession.inputNames;
    this.outputNames = this.#inferenceSession.outputNames;
  }

  dispose(): Promise<void> {
    return Promise.resolve();
  }

  readonly inputNames: string[];
  readonly outputNames: string[];


  run(feeds: SessionHandler.FeedsType, fetches: SessionHandler.FetchesType,
      options: InferenceSession.RunOptions): Promise<SessionHandler.ReturnType> {
    return new Promise((resolve, reject) => {
      process.nextTick(() => {
        try {
          resolve(this.#inferenceSession.run(feeds, fetches, options));
        } catch (e) {
          // reject if any error is thrown
          reject(e);
        }
      });
    });
  }
}

class OnnxruntimeBackend implements Backend {
  init(): Promise<void> {
    return Promise.resolve();
  }

  createSessionHandler(path: string, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  createSessionHandler(buffer: Uint8Array, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  createSessionHandler(pathOrBuffer: string|Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<SessionHandler> {
    return new Promise((resolve, reject) => {
      process.nextTick(() => {
        try {
          resolve(new OnnxruntimeSessionHandler(pathOrBuffer, options || {}));
        } catch (e) {
          // reject if any error is thrown
          reject(e);
        }
      });
    });
  }
}

export const onnxruntimeBackend = new OnnxruntimeBackend();
