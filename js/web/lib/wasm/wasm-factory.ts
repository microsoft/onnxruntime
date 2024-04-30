// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Env} from 'onnxruntime-common';

import type {OrtWasmModule} from './wasm-types';
import {dynamicImportDefault} from './wasm-utils-import';

let wasm: OrtWasmModule|undefined;
let initialized = false;
let initializing = false;
let aborted = false;

const isMultiThreadSupported = (): boolean => {
  // If 'SharedArrayBuffer' is not available, WebAssembly threads will not work.
  if (typeof SharedArrayBuffer === 'undefined') {
    return false;
  }

  try {
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
  let numThreads = flags.numThreads!;

  // ensure SIMD is supported
  if (!isSimdSupported()) {
    throw new Error('WebAssembly SIMD is not supported in the current environment.');
  }

  // check if multi-threading is supported
  const multiThreadSupported = isMultiThreadSupported();
  if (numThreads > 1 && !multiThreadSupported) {
    if (typeof self !== 'undefined' && !self.crossOriginIsolated) {
      // eslint-disable-next-line no-console
      console.warn(
          'env.wasm.numThreads is set to ' + numThreads +
          ', but this will not work unless you enable crossOriginIsolated mode. ' +
          'See https://web.dev/cross-origin-isolation-guide/ for more info.');
    }

    // eslint-disable-next-line no-console
    console.warn(
        'WebAssembly multi-threading is not supported in the current environment. ' +
        'Falling back to single-threading.');

    // set flags.numThreads to 1 so that OrtInit() will not create a global thread pool.
    flags.numThreads = numThreads = 1;
  }

  const wasmPaths = flags.wasmPaths;
  const wasmPrefixOverride = typeof wasmPaths === 'string' ? wasmPaths : undefined;
  const wasmFileName = !BUILD_DEFS.DISABLE_TRAINING ? 'ort-training-wasm-simd-threaded' :
      !BUILD_DEFS.DISABLE_JSEP                      ? 'ort-wasm-simd-threaded.jsep' :
                                                      'ort-wasm-simd-threaded';
  const wasmPathOverride = typeof wasmPaths === 'object' ? wasmPaths[`${wasmFileName}.wasm`] : undefined;
  const wasmOverride =
      wasmPathOverride ?? (wasmPrefixOverride ? wasmPrefixOverride + wasmFileName + '.wasm' : undefined);

  const [objectUrl, ortWasmFactory] = (await dynamicImportDefault<EmscriptenModuleFactory<OrtWasmModule>>(
      `${wasmFileName}.mjs`, wasmPrefixOverride, numThreads === 1));

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
    const config: Partial<OrtWasmModule> = {
      numThreads,
      locateFile: ((fileName, scriptDirectory) => wasmOverride ?? scriptDirectory + fileName)
    };

    ortWasmFactory(config).then(
        // wasm module initialized successfully
        module => {
          initializing = false;
          initialized = true;
          wasm = module;
          resolve();
          if (objectUrl) {
            URL.revokeObjectURL(objectUrl);
          }
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

    wasm?.PThread?.terminateAllThreads();
    wasm = undefined;

    initializing = false;
    initialized = false;
    aborted = true;
  }
};
