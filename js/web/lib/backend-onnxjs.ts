/* eslint-disable import/no-internal-modules */
import {Backend, env, InferenceSession, SessionHandler} from 'onnxruntime-common';
import {Session} from './onnxjs/session';
import {OnnxjsSessionHandler} from './onnxjs/session-handler';

class OnnxjsBackend implements Backend {
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  async init(): Promise<void> {}

  async createSessionHandler(pathOrBuffer: string|Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<SessionHandler> {
    // NOTE: Session.Config(from onnx.js) is not compatible with InferenceSession.SessionOptions(from
    // onnxruntime-common).
    //       In future we should remove Session.Config and use InferenceSession.SessionOptions.
    //       Currently we allow this to happen to make test runner work.
    const session = new Session(options as unknown as Session.Config);

    // typescript cannot merge method override correctly (so far in 4.2.3). need if-else to call the method.
    if (typeof pathOrBuffer === 'string') {
      await session.loadModel(pathOrBuffer);
    } else {
      await session.loadModel(pathOrBuffer);
    }

    return new OnnxjsSessionHandler(session);
  }
}

export const onnxjsBackend = new OnnxjsBackend();

export interface WebGLFlags {
  /**
   * set or get the WebGL Context ID (webgl or webgl2)
   */
  contextId?: 'webgl'|'webgl2';
  /**
   * set or get the maximum batch size for matmul. 0 means to disable batching.
   */
  matmulMaxBatchSize?: number;
  /**
   * set or get the texture cache mode
   */
  textureCacheMode?: 'initializerOnly'|'full';
  /**
   * set or get the packed texture mode
   */
  pack?: boolean;
}

/**
 * Represent a set of flags for ONNX.js backend.
 */
export const flags: WebGLFlags = env.webgl = env.webgl as WebGLFlags || {};
