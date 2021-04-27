// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import wasmModuleFactory, {BackendWasmModule} from './onnxruntime_wasm';

// some global parameters to deal with wasm binding initialization
let wasm: BackendWasmModule;
let initialized = false;
let initializing = false;

/**
 * initialize the WASM instance.
 *
 * this function should be called before any other calls to the WASM binding.
 */
export const init = async(): Promise<void> => {
  if (initialized) {
    return Promise.resolve();
  }
  if (initializing) {
    throw new Error('multiple calls to \'init()\' detected.');
  }

  initializing = true;

  return new Promise<void>((resolve, reject) => {
    wasmModuleFactory().then(
        initializedModule => {
          // resolve init() promise
          wasm = initializedModule;
          initializing = false;
          initialized = true;
          resolve();
        },
        err => {
          initializing = false;
          reject(err);
        });
  });
};

export const getInstance = (): BackendWasmModule => wasm;
