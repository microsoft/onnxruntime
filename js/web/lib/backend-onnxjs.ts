import {Backend, env, InferenceSession, SessionHandler} from 'onnxruntime-common';

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
