// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {EnvIml} from './env-impl';
export declare namespace Env {
  export interface WebAssemblyFlags {
    /**
     * set or get number of thread(s). If omitted or set to 0, number of thread(s) will be determined by system. If set
     * to 1, no worker thread will be spawned.
     *
     * This setting is available only when WebAssembly multithread feature is available in current context.
     */
    numThreads?: number;

    /**
     * Set or get a number specifying the timeout for initialization of WebAssembly backend, in milliseconds. A zero
     * value indicates no timeout is set. (default is 0)
     */
    initTimeout?: number;
  }

  export interface WebGLFlags {
    /**
     * Set or get the WebGL Context ID (webgl or webgl2). Default value is webgl2.
     */
    contextId?: 'webgl'|'webgl2';
    /**
     * Set or get the maximum batch size for matmul. 0 means to disable batching.
     */
    matmulMaxBatchSize?: number;
    /**
     * Set or get the texture cache mode. Default value is full.
     */
    textureCacheMode?: 'initializerOnly'|'full';
    /**
     * Set or get the packed texture mode
     */
    pack?: boolean;
  }
}

export interface Env {
  /**
   * set a logging level. If omitted, default is 'warning'
   */
  loggingLevel?: 'verbose'|'info'|'warning'|'error'|'fatal';
  /**
   * Indicate whether run in debug mode.
   */
  debug?: boolean;

  /**
   * Represent a set of flags for WebAssembly
   */
  wasm: Env.WebAssemblyFlags;

  /**
   * Represent a set of flags for WebGL
   */
  webgl: Env.WebGLFlags;

  [name: string]: unknown;
}

/**
 * Represent a set of flags as a global singleton.
 */
export const env: EnvIml = {
  wasm: {},
  webgl: {},
  loggingLevel: 'warning',
};
