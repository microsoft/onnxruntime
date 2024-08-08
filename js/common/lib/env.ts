// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {env as envImpl} from './env-impl.js';
import {TryGetTypeIfDeclared} from './type-helper.js';

export declare namespace Env {
  export type WasmPathPrefix = string;
  export interface WasmFilePaths {
    /**
     * Specify the override path for the main .wasm file.
     *
     * This path should be an absolute path.
     *
     * If not modified, the filename of the .wasm file is:
     * - `ort-wasm-simd-threaded.wasm` for default build
     * - `ort-wasm-simd-threaded.jsep.wasm` for JSEP build (with WebGPU and WebNN)
     * - `ort-training-wasm-simd-threaded.wasm` for training build
     */
    wasm?: URL|string;
    /**
     * Specify the override path for the main .mjs file.
     *
     * This path should be an absolute path.
     *
     * If not modified, the filename of the .mjs file is:
     * - `ort-wasm-simd-threaded.mjs` for default build
     * - `ort-wasm-simd-threaded.jsep.mjs` for JSEP build (with WebGPU and WebNN)
     * - `ort-training-wasm-simd-threaded.mjs` for training build
     */
    mjs?: URL|string;
  }
  export type WasmPrefixOrFilePaths = WasmPathPrefix|WasmFilePaths;
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
     * @deprecated This property is deprecated. Since SIMD is supported by all major JavaScript engines, non-SIMD
     * build is no longer provided. This property will be removed in future release.
     * @defaultValue `true`
     */
    simd?: boolean;

    /**
     * set or get a boolean value indicating whether to enable trace.
     *
     * @deprecated Use `env.trace` instead. If `env.trace` is set, this property will be ignored.
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
     * Set a custom URL prefix to the .wasm/.mjs files, or an object of overrides for both .wasm/.mjs file. The override
     * path should be an absolute path.
     */
    wasmPaths?: WasmPrefixOrFilePaths;

    /**
     * Set a custom buffer which contains the WebAssembly binary. If this property is set, the `wasmPaths` property will
     * be ignored.
     */
    wasmBinary?: ArrayBufferLike|Uint8Array;

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
    programName: string;
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
    profiling: {
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
     * @deprecated Create your own GPUAdapter instance and set {@link adapter} property if you want to use an WebGPU
     * adapter with specific power preference.
     *
     * Set or get the power preference.
     *
     * Setting this property only has effect before the first WebGPU inference session is created. The value will be
     * used as options for `navigator.gpu.requestAdapter()`.
     *
     * See {@link https://gpuweb.github.io/gpuweb/#dictdef-gpurequestadapteroptions} for more details.
     *
     * @defaultValue `undefined`
     */
    powerPreference?: 'low-power'|'high-performance';
    /**
     * @deprecated Create your own GPUAdapter instance and set property {@link adapter} if you want to use an WebGPU
     * adapter with force fallback adapter.
     *
     * Set or get the force fallback adapter flag.
     *
     * Setting this property only has effect before the first WebGPU inference session is created. The value will be
     * used as options for `navigator.gpu.requestAdapter()`.
     *
     * See {@link https://gpuweb.github.io/gpuweb/#dictdef-gpurequestadapteroptions} for more details.
     *
     * @defaultValue `undefined`
     */
    forceFallbackAdapter?: boolean;
    /**
     * Set the GPU adapter for WebGPU. The value will be used for the underlying WebGPU backend to create GPU device.
     *
     * Setting this property only has effect before either property {@link device} has been accessed or the first WebGPU
     * inference session is created.
     *
     * When setting this property, the value should be a `GPUAdapter` object. If the value is not a `GPUAdapter` object,
     * an error will be thrown.
     *
     * If this property is not set, the WebGPU backend will create a new GPU adapter just before creating the first
     * WebGPU inference session.
     */
    set adapter(value: TryGetTypeIfDeclared<'GPUAdapter'>);
    /**
     * Set or get the GPU device for WebGPU.
     *
     * There are several scenarios of accessing this property:
     * - Set a value before the first WebGPU inference session is created. The value will be used by the WebGPU backend
     * to perform calculations. This operation can only be done once.
     * - Set a value after the first WebGPU inference session is created. This operation is invalid and will cause an
     * error to be thrown.
     * - Get the value before the first WebGPU inference session is created. This will try to create a new GPUDevice
     * instance. Use {@link adapter} if set or create a new GPU adapter to create a new GPU device. A `Promise` that
     * resolves to a `GPUDevice` object will be returned. This operation can only be done once.
     * - Get the value after the first WebGPU inference session is created. This will return a resolved `Promise` to the
     * `GPUDevice` object used by the WebGPU backend. This operation can be done multiple times.
     *
     * When setting this property, the value should be a `GPUDevice` object. If the value is not a `GPUDevice` object,
     * an error will be thrown.
     *
     * When getting this property, the value will be a `Promise` that resolves to a `GPUDevice` object.
     *
     * If this property is neither set nor get, the WebGPU backend will create a new GPU device just before creating the
     * first WebGPU inference session. You can still get the value after that.
     */
    get device(): Promise<TryGetTypeIfDeclared<'GPUDevice'>>;
    set device(value: TryGetTypeIfDeclared<'GPUDevice'>);
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
   * set or get a boolean value indicating whether to enable trace.
   *
   * @defaultValue `false`
   */
  trace?: boolean;

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
