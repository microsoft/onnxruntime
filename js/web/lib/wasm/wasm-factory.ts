// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Env} from 'onnxruntime-common';
import * as path from 'path';

import {OrtWasmModule} from './binding/ort-wasm';
import {OrtWasmThreadedModule} from './binding/ort-wasm-threaded';

/* eslint-disable @typescript-eslint/no-require-imports */
const ortWasmFactory: EmscriptenModuleFactory<OrtWasmModule> =
    BUILD_DEFS.DISABLE_WEBGPU ? require('./binding/ort-wasm.js') : require('./binding/ort-wasm-simd.jsep.js');

const ortWasmFactoryThreaded: EmscriptenModuleFactory<OrtWasmModule> = !BUILD_DEFS.DISABLE_WASM_THREAD ?
    (BUILD_DEFS.DISABLE_WEBGPU ? require('./binding/ort-wasm-threaded.js') :
                                 require('./binding/ort-wasm-simd-threaded.jsep.js')) :
    ortWasmFactory;
/* eslint-enable @typescript-eslint/no-require-imports */

let wasm: OrtWasmModule|undefined;
let initialized = false;
let initializing = false;
let aborted = false;

const isMultiThreadSupported = (): boolean => {
  try {
    // If 'SharedArrayBuffer' is not available, WebAssembly threads will not work.
    if (typeof SharedArrayBuffer === 'undefined') {
      return false;
    }

    // Test for transferability of SABs (for browsers. needed for Firefox)
    // https://groups.google.com/forum/#!msg/mozilla.dev.platform/IHkBZlHETpA/dwsMNchWEQAJ
    if (typeof MessageChannel !== 'undefined') {
      new MessageChannel().port1.postMessage(new SharedArrayBuffer(1));
    }

    // Test for WebAssembly threads capability (for both browsers and Node.js)
    // This typed array is a WebAssembly program containing threaded instructions.
    return WebAssembly.validate(new Uint8Array([
      0, 97, 115, 109, 1, 0,  0,  0, 1, 4, 1,  96, 0,   0,  3, 2, 1,  0, 5,
      4, 1,  3,   1,   1, 10, 11, 1, 9, 0, 65, 0,  254, 16, 2, 0, 26, 11
    ]));
  } catch (e) {
    return false;
  }
};

const isSimdSupported = (): boolean => {
  try {
    // Test for WebAssembly SIMD capability (for both browsers and Node.js)
    // This typed array is a WebAssembly program containing SIMD instructions.

    // The binary data is generated from the following code by wat2wasm:
    //
    // (module
    //   (type $t0 (func))
    //   (func $f0 (type $t0)
    //     (drop
    //       (i32x4.dot_i16x8_s
    //         (i8x16.splat
    //           (i32.const 0))
    //         (v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000)))))

    return WebAssembly.validate(new Uint8Array([
      0,   97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 30, 1,   28,  0, 65, 0,
      253, 15, 253, 12,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0,  253, 186, 1, 26, 11
    ]));
  } catch (e) {
    return false;
  }
};

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

  const useThreads = numThreads > 1 && isMultiThreadSupported();
  const useSimd = simd && isSimdSupported();

  const wasmPaths = flags.wasmPaths;
  const wasmPrefixOverride = typeof wasmPaths === 'string' ? wasmPaths : undefined;
  const wasmFileName = getWasmFileName(useSimd, useThreads);
  const wasmPathOverride = typeof wasmPaths === 'object' ? wasmPaths[wasmFileName] : undefined;

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

        if (fileName.endsWith('.wasm')) {
          if (wasmPathOverride) {
            return wasmPathOverride;
          }

          const prefix = wasmPrefixOverride ?? scriptDirectory;

          if (!BUILD_DEFS.DISABLE_WEBGPU) {
            if (wasmFileName === 'ort-wasm-simd.wasm') {
              return prefix + 'ort-wasm-simd.jsep.wasm';
            } else if (wasmFileName === 'ort-wasm-simd-threaded.wasm') {
              return prefix + 'ort-wasm-simd-threaded.jsep.wasm';
            }
          }

          return prefix + wasmFileName;
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
