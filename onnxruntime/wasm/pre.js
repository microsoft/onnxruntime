// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

/**
 * Mount external data files of a model to an internal map, which will be used during session initialization.
 *
 * @param {string} externalDataFilesPath
 * @param {Uint8Array} externalDataFilesData
 */
Module['mountExternalData'] = (externalDataFilePath, externalDataFileData) => {
  const files = Module.MountedFiles || (Module.MountedFiles = new Map());
  files.set(externalDataFilePath, externalDataFileData);
};

/**
 * Unmount external data files of a model.
 */
Module['unmountExternalData'] = () => {
  delete Module.MountedFiles;
};

/**
 * A workaround for SharedArrayBuffer when it is not available in the current context.
 *
 * We need this workaround because Emscripten generates code that assumes `SharedArrayBuffer` is always available and
 * uses SharedArrayBuffer in this way:
 * ```js
 * buffer instanceof SharedArrayBuffer
 * ```
 *
 * This code will throw an error when SharedArrayBuffer is not available. Fortunately, we can use `WebAssembly.Memory`
 * to create an instance of SharedArrayBuffer even when SharedArrayBuffer is not available in `globalThis`.
 *
 * While this workaround allows the WebAssembly module to be loaded, it does not provide multi-threading features when
 * SharedArrayBuffer is not available in `globalThis`. The WebAssembly module will run well in a single thread, when:
 * - Module['numThreads'] is set to 1, and
 * - _OrtInit() is called with numThreads = 1.
 *
 * @suppress {checkVars}
 */
var SharedArrayBuffer = globalThis.SharedArrayBuffer ??
    new WebAssembly.Memory({'initial': 0, 'maximum': 0, 'shared': true}).buffer.constructor;

/**
 * Allow to override the path of the WebAssembly file.
 *
 * The path can be overridden by setting a global variable `wasmOverride`. (injected when loaded as Blob)
 *
 * We need this workaround because since Emscripten 3.1.58, the <filename>.worker.[m]js file is not generated anymore.
 * Instead, the <filename>.[m]js will create a Worker using its own URL. This will not work when the script is loaded
 * from a Blob URL because the Blob URL loses the relative path information. Also, this behavior does not honor the
 * `wasmPaths` override.
 *
 * When loaded as a Blob, we append a line to the beginning of the script to write a `wasmOverride` variable and assign
 * the URL/path of the .wasm file to it. The `locateFile` function will use this variable to locate the .wasm file if
 * it is defined.
 */
if (!Module['locateFile'] && typeof wasmOverride !== 'undefined') {
  Module['locateFile'] = () => wasmOverride;
}
