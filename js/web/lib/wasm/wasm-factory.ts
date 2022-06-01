// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Env} from 'onnxruntime-common';
import * as path from 'path';
import * as wasmFeatureDetect from 'wasm-feature-detect';

import {OrtWasmModule} from './binding/ort-wasm';
import {OrtWasmThreadedModule} from './binding/ort-wasm-threaded';
import ortWasmFactory from './binding/ort-wasm.js';

const ortWasmFactoryThreaded: EmscriptenModuleFactory<OrtWasmModule> =
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    !BUILD_DEFS.DISABLE_WASM_THREAD ? require('./binding/ort-wasm-threaded.js') : ortWasmFactory;

let wasm: OrtWasmModule|undefined;
let initialized = false;
let initializing = false;
let aborted = false;

const getWasmFileName = (useSimd: boolean, useThreads: boolean) => {
  if (useThreads) {
    return useSimd ? 'ort-wasm-simd-threaded.wasm' : 'ort-wasm-threaded.wasm';
  } else {
    return useSimd ? 'ort-wasm-simd.wasm' : 'ort-wasm.wasm';
  }
};

export const initializeWebAssembly = async(flags: Env.WebAssemblyFlags): Promise<void> => {
  if (initialized) {
    return Promise.resolve();
  }
  if (initializing) {
    throw new Error('multiple calls to \'initializeWebAssembly()\' detected.');
  }
  if (aborted) {
    throw new Error('previous call to \'initializeWebAssembly()\' failed.');
  }

  initializing = true;

  // wasm flags are already initialized
  const timeout = flags.initTimeout!;
  const numThreads = flags.numThreads!;
  const simd = flags.simd!;

  const useThreads = numThreads > 1 && await wasmFeatureDetect.threads();
  const useSimd = simd && await wasmFeatureDetect.simd();

  const wasmPrefixOverride = typeof flags.wasmPaths === 'string' ? flags.wasmPaths : undefined;
  const wasmFileName = getWasmFileName(false, useThreads);
  const wasmOverrideFileName = getWasmFileName(useSimd, useThreads);
  const wasmPathOverride = typeof flags.wasmPaths === 'object' ? flags.wasmPaths[wasmOverrideFileName] : undefined;

  let isTimeout = false;

  const tasks: Array<Promise<void>> = [];

  // promise for timeout
  if (timeout > 0) {
    tasks.push(new Promise((resolve) => {
      setTimeout(() => {
        isTimeout = true;
        resolve();
      }, timeout);
    }));
  }

  // promise for module initialization
  tasks.push(new Promise((resolve, reject) => {
    const factory = useThreads ? ortWasmFactoryThreaded : ortWasmFactory;
    const config: Partial<OrtWasmModule> = {
      locateFile: (fileName: string, scriptDirectory: string) => {
        if (!BUILD_DEFS.DISABLE_WASM_THREAD && useThreads && fileName.endsWith('.worker.js') &&
            typeof Blob !== 'undefined') {
          return URL.createObjectURL(new Blob(
              [
                // This require() function is handled by webpack to load file content of the corresponding .worker.js
                // eslint-disable-next-line @typescript-eslint/no-require-imports
                require('./binding/ort-wasm-threaded.worker.js')
              ],
              {type: 'text/javascript'}));
        }

        if (fileName === wasmFileName) {
          const prefix: string = wasmPrefixOverride ?? scriptDirectory;
          return wasmPathOverride ?? prefix + wasmOverrideFileName;
        }

        return scriptDirectory + fileName;
      }
    };

    if (!BUILD_DEFS.DISABLE_WASM_THREAD && useThreads) {
      if (typeof Blob === 'undefined') {
        config.mainScriptUrlOrBlob = path.join(__dirname, 'ort-wasm-threaded.js');
      } else {
        const scriptSourceCode = `var ortWasmThreaded=(function(){var _scriptDir;return ${factory.toString()}})();`;
        config.mainScriptUrlOrBlob = new Blob([scriptSourceCode], {type: 'text/javascript'});
      }
    }

    factory(config).then(
        // wasm module initialized successfully
        module => {
          initializing = false;
          initialized = true;
          wasm = module;
          resolve();
        },
        // wasm module failed to initialize
        (what) => {
          initializing = false;
          aborted = true;
          reject(what);
        });
  }));

  await Promise.race(tasks);

  if (isTimeout) {
    throw new Error(`WebAssembly backend initializing failed due to timeout: ${timeout}ms`);
  }
};

export const getInstance = (): OrtWasmModule => {
  if (initialized && wasm) {
    return wasm;
  }

  throw new Error('WebAssembly is not initialized yet.');
};

export const dispose = (): void => {
  if (initialized && !initializing && !aborted) {
    initializing = true;

    (wasm as OrtWasmThreadedModule).PThread?.terminateAllThreads();
    wasm = undefined;

    initializing = false;
    initialized = false;
    aborted = true;
  }
};
