// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Backend, env, InferenceSession, InferenceSessionHandler} from 'onnxruntime-common';

import {initializeOrtEp, initializeWebAssemblyAndOrtRuntime} from './wasm/proxy-wrapper';
import {OnnxruntimeWebAssemblySessionHandler} from './wasm/session-handler-inference';
import {scriptSrc} from './wasm/wasm-utils-import';

/**
 * This function initializes all flags for WebAssembly.
 *
 * Those flags are accessible from `ort.env.wasm`. Users are allow to set those flags before the first inference session
 * being created, to override default value.
 */
export const initializeFlags = (): void => {
  if (typeof env.wasm.initTimeout !== 'number' || env.wasm.initTimeout < 0) {
    env.wasm.initTimeout = 0;
  }

  if (env.wasm.simd === false) {
    // eslint-disable-next-line no-console
    console.warn(
        'Deprecated property "env.wasm.simd" is set to false. ' +
        'non-SIMD build is no longer provided, and this setting will be ignored.');
  }

  if (typeof env.wasm.proxy !== 'boolean') {
    env.wasm.proxy = false;
  }

  if (typeof env.wasm.trace !== 'boolean') {
    env.wasm.trace = false;
  }

  if (typeof env.wasm.numThreads !== 'number' || !Number.isInteger(env.wasm.numThreads) || env.wasm.numThreads <= 0) {
    // The following logic only applies when `ort.env.wasm.numThreads` is not set by user. We will always honor user's
    // setting if it is provided.

    // Browser: when crossOriginIsolated is false, SharedArrayBuffer is not available so WebAssembly threads will not
    // work. In this case, we will set numThreads to 1.
    //
    // There is an exception: when the browser is configured to force-enable SharedArrayBuffer (e.g. Chromuim with
    // --enable-features=SharedArrayBuffer), it is possible that `self.crossOriginIsolated` is false and
    // SharedArrayBuffer is available at the same time. This is usually for testing. In this case,  we will still set
    // numThreads to 1 here. If we want to enable multi-threading in test, we should set `ort.env.wasm.numThreads` to a
    // value greater than 1.
    if (typeof self !== 'undefined' && !self.crossOriginIsolated) {
      env.wasm.numThreads = 1;
    } else {
      const numCpuLogicalCores =
          typeof navigator === 'undefined' ? require('node:os').cpus().length : navigator.hardwareConcurrency;
      env.wasm.numThreads = Math.min(4, Math.ceil((numCpuLogicalCores || 1) / 2));
    }
  }

  if (!BUILD_DEFS.DISABLE_DYNAMIC_IMPORT) {
    // overwrite wasm paths override if not set
    if (env.wasm.wasmPaths === undefined && scriptSrc && scriptSrc.indexOf('blob:') !== 0) {
      env.wasm.wasmPaths = scriptSrc.substring(0, scriptSrc.lastIndexOf('/') + 1);
    }
  }
};

export class OnnxruntimeWebAssemblyBackend implements Backend {
  /**
   * This function initializes the WebAssembly backend.
   *
   * This function will be called only once for each backend name. It will be called the first time when
   * `ort.InferenceSession.create()` is called with a registered backend name.
   *
   * @param backendName - the registered backend name.
   */
  async init(backendName: string): Promise<void> {
    // populate wasm flags
    initializeFlags();

    // init wasm
    await initializeWebAssemblyAndOrtRuntime();

    // performe EP specific initialization
    await initializeOrtEp(backendName);
  }
  createInferenceSessionHandler(path: string, options?: InferenceSession.SessionOptions):
      Promise<InferenceSessionHandler>;
  createInferenceSessionHandler(buffer: Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<InferenceSessionHandler>;
  async createInferenceSessionHandler(pathOrBuffer: string|Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<InferenceSessionHandler> {
    const handler = new OnnxruntimeWebAssemblySessionHandler();
    await handler.loadModel(pathOrBuffer, options);
    return Promise.resolve(handler);
  }
}
