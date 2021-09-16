// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {EnvImpl} from './env-impl';
export declare namespace Env {
  export type WasmPrefixOrFilePaths = string|{
    'ort-wasm.wasm'?: string;
    'ort-wasm-threaded.wasm'?: string;
    'ort-wasm-simd.wasm'?: string;
    'ort-wasm-simd-threaded.wasm'?: string;
  };
  export interface WebAssemblyFlags {
    /**
     * set or get number of thread(s). If omitted or set to 0, number of thread(s) will be determined by system. If set
     * to 1, no worker thread will be spawned.
     *
     * This setting is available only when WebAssembly multithread feature is available in current context.
     *
     * @defaultValue `0`
     */
    numThreads?: number;

    /**
     * set or get a boolean value indicating whether to enable SIMD. If set to false, SIMD will be forcely disabled.
     *
     * This setting is available only when WebAssembly SIMD feature is available in current context.
     *
     * @defaultValue `true`
     */
    simd?: boolean;

    /**
     * Set or get a number specifying the timeout for initialization of WebAssembly backend, in milliseconds. A zero
     * value indicates no timeout is set.
     *
     * @defaultValue `0`
     */
    initTimeout?: number;

    /**
     * Set a custom URL prefix to the .wasm files or a set of overrides for each .wasm file. The override path should be
     * an absolute path.
     */
    wasmPaths?: WasmPrefixOrFilePaths;

    /**
     * Set or get a boolean value indicating whether to proxy the execution of main thread to a worker thread.
     *
     * @defaultValue `false`
     */
    proxy?: boolean;
  }

  export interface WebGLFlags {
    /**
     * Set or get the WebGL Context ID (webgl or webgl2).
     *
     * @defaultValue `'webgl2'`
     */
    contextId?: 'webgl'|'webgl2';
    /**
     * Set or get the maximum batch size for matmul. 0 means to disable batching.
     *
     * @deprecated
     */
    matmulMaxBatchSize?: number;
    /**
     * Set or get the texture cache mode.
     *
     * @defaultValue `'full'`
     */
    textureCacheMode?: 'initializerOnly'|'full';
    /**
     * Set or get the packed texture mode
     *
     * @defaultValue `false`
     */
    pack?: boolean;
    /**
     * Set or get whether enable async download.
     *
     * @defaultValue `false`
     */
    async?: boolean;
  }
}

export interface Env {
  /**
   * set the severity level for logging.
   *
   * @defaultValue `'warning'`
   */
  logLevel?: 'verbose'|'info'|'warning'|'error'|'fatal';
  /**
   * Indicate whether run in debug mode.
   *
   * @defaultValue `false`
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
export const env: Env = new EnvImpl();
