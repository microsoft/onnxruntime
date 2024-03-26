// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import type {OrtWasmModule} from '../ort-wasm.js';

export const esmLoaderImport = async(useThreads: boolean): Promise<EmscriptenModuleFactory<OrtWasmModule>> => {
  if (BUILD_DEFS.DISABLE_WASM_THREAD || !useThreads) {
    if (!BUILD_DEFS.DISABLE_TRAINING) {
      return (await import('../ort-training-wasm-simd.js')) as unknown as EmscriptenModuleFactory<OrtWasmModule>;
    } else {
      return BUILD_DEFS.DISABLE_WEBGPU ?
          (await import('../ort-wasm.js')) as unknown as EmscriptenModuleFactory<OrtWasmModule>:
          (await import('../ort-wasm-simd.jsep.js')) as unknown as EmscriptenModuleFactory<OrtWasmModule>;
    }
  } else {
    return BUILD_DEFS.DISABLE_WEBGPU ?
        (await import('../ort-wasm-threaded.js')) as unknown as EmscriptenModuleFactory<OrtWasmModule>:
        (await import('../ort-wasm-simd-threaded.jsep.js')) as unknown as EmscriptenModuleFactory<OrtWasmModule>;
  }
};
