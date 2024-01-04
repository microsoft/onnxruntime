// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {env as envImpl} from './env-impl.js';

export declare namespace Env {
  export type WasmPrefixOrFilePaths = string|{
    /* eslint-disable @typescript-eslint/naming-convention */
    'ort-wasm.wasm'?: string;
    'ort-wasm-threaded.wasm'?: string;
    'ort-wasm-simd.wasm'?: string;
    'ort-training-wasm-simd.wasm'?: string;
    'ort-wasm-simd-threaded.wasm'?: string;
    /* eslint-enable @typescript-eslint/naming-convention */
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
     * set or get a boolean value indicating whether to enable trace.
     *
     * @defaultValue `false`
     */
    trace?: boolean;

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
     * Get the WebGL rendering context.
     */
    readonly context: WebGLRenderingContext;
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

  export interface WebGpuProfilingDataV1TensorMetadata {
    dims: readonly number[];
    dataType: string;
  }
  export interface WebGpuProfilingDataV1 {
    version: 1;
    inputsMetadata: readonly WebGpuProfilingDataV1TensorMetadata[];
    outputsMetadata: readonly WebGpuProfilingDataV1TensorMetadata[];
    kernelId: number;
    kernelType: string;
    kernelName: string;
    startTime: number;
    endTime: number;
  }

  export type WebGpuProfilingData = WebGpuProfilingDataV1;

  export interface WebGpuFlags {
    /**
     * Set or get the profiling mode.
     *
     * @deprecated Use `env.webgpu.profiling.mode` instead. If `env.webgpu.profiling.mode` is set, this property will be
     * ignored.
     */
    profilingMode?: 'off'|'default';
    /**
     * Set or get the profiling configuration.
     */
    profiling?: {
      /**
       * Set or get the profiling mode.
       *
       * @defaultValue `'off'`
       */
      mode?: 'off'|'default';

      /**
       * Set or get a callback function when a profiling data is received. If not set, the profiling data will be
       * printed to console.
       */
      ondata?: (data: WebGpuProfilingData) => void;
    };
    /**
     * Get the device for WebGPU.
     *
     * When use with TypeScript, the type of this property is `GPUDevice` defined in "@webgpu/types".
     * Use `const device = env.webgpu.device as GPUDevice;` in TypeScript to access this property with correct type.
     *
     * see comments on {@link GpuBufferType} for more details about why not use types defined in "@webgpu/types".
     */
    readonly device: unknown;
    /**
     * Set or get whether validate input content.
     *
     * @defaultValue `false`
     */
    validateInputContent?: boolean;
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
   * Get version of the current package.
   */
  readonly versions: {
    readonly common: string;
    readonly web?: string;
    readonly node?: string;
    // eslint-disable-next-line @typescript-eslint/naming-convention
    readonly 'react-native'?: string;
  };

  /**
   * Represent a set of flags for WebAssembly
   */
  readonly wasm: Env.WebAssemblyFlags;

  /**
   * Represent a set of flags for WebGL
   */
  readonly webgl: Env.WebGLFlags;

  /**
   * Represent a set of flags for WebGPU
   */
  readonly webgpu: Env.WebGpuFlags;

  [name: string]: unknown;
}

/**
 * Represent a set of flags as a global singleton.
 */
export const env: Env = envImpl;
