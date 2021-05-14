// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {env} from 'onnxruntime-common';
import * as path from 'path';

import {OrtWasmModule} from './binding/ort-wasm';
import {OrtWasmThreadedModule} from './binding/ort-wasm-threaded';
import ortWasmFactoryThreaded from './binding/ort-wasm-threaded.js';
import ortWasmFactory from './binding/ort-wasm.js';

let wasm: OrtWasmModule;
let initialized = false;
let initializing = false;
let aborted = false;

const isMultiThreadSupported = (): boolean => {
  try {
    // Test for transferability of SABs (needed for Firefox)
    // https://groups.google.com/forum/#!msg/mozilla.dev.platform/IHkBZlHETpA/dwsMNchWEQAJ
    if (typeof MessageChannel !== 'undefined') {
      new MessageChannel().port1.postMessage(new SharedArrayBuffer(1));
    }
    // This typed array is a WebAssembly program containing threaded
    // instructions.
    return WebAssembly.validate(new Uint8Array([
      0, 97, 115, 109, 1, 0,  0,  0, 1, 4, 1,  96, 0,   0,  3, 2, 1,  0, 5,
      4, 1,  3,   1,   1, 10, 11, 1, 9, 0, 65, 0,  254, 16, 2, 0, 26, 11
    ]));
  } catch (e) {
    return false;
  }
};

export const initializeWebAssembly = async(): Promise<void> => {
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
  const timeout = env.wasm.initTimeout!;
  const numThreads = env.wasm.numThreads!;

  const useThreads = numThreads > 1 && isMultiThreadSupported();
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
    const config: Partial<OrtWasmModule> = {};

    if (useThreads) {
      if (typeof Blob === 'undefined') {
        config.mainScriptUrlOrBlob = path.join(__dirname, 'ort-wasm-threaded.js');
      } else {
        const scriptSourceCode =
            `var ortWasmThreaded=(function(){var _scriptDir;return ${ortWasmFactoryThreaded.toString()}})();`;
        config.mainScriptUrlOrBlob = new Blob([scriptSourceCode], {type: 'text/javascript'});
        config.locateFile = (fileName: string, scriptDirectory: string) => {
          if (fileName.endsWith('.worker.js')) {
            return URL.createObjectURL(new Blob(
                [
                  // This require() function is handled by webpack to load file content of the corresponding .worker.js
                  // eslint-disable-next-line @typescript-eslint/no-require-imports
                  require('./binding/ort-wasm-threaded.worker.js')
                ],
                {type: 'text/javascript'}));
          }
          return scriptDirectory + fileName;
        };
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
  if (initialized) {
    return wasm;
  }

  throw new Error('WebAssembly is not initialized yet.');
};

export const dispose = (): void => {
  if (initialized && !initializing && !aborted) {
    initializing = true;

    (wasm as OrtWasmThreadedModule).PThread?.terminateAllThreads();

    initializing = false;
    initialized = false;
    aborted = true;
  }
};
