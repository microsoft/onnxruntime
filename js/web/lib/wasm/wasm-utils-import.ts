// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import type {OrtWasmModule} from './wasm-types';
import {isNode} from './wasm-utils-env';

/**
 * The classic script source URL. This is not always available in non ESModule environments.
 *
 * In Node.js, this is undefined.
 */
export const scriptSrc =
    // if Nodejs, return undefined
    isNode ? undefined :
             // if It's ESM, use import.meta.url
             BUILD_DEFS.ESM_IMPORT_META_URL ??
        // use `document.currentScript.src` if available
        (typeof document !== 'undefined' ? (document.currentScript as HTMLScriptElement)?.src :
                                           // use `self.location.href` if available
                                           (typeof self !== 'undefined' ? self.location?.href : undefined));

/**
 * The origin of the current location.
 *
 * In Node.js, this is undefined.
 */
const origin = isNode || typeof location === 'undefined' ? undefined : location.origin;

/**
 * Check if the given filename with prefix is from the same origin.
 */
const isSameOrigin = (filename: string, prefixOverride?: string) => {
  try {
    const baseUrl = prefixOverride ?? scriptSrc;
    const url = baseUrl ? new URL(filename, baseUrl) : new URL(filename);
    return url.origin === origin;
  } catch {
    return false;
  }
};

/**
 * Normalize the inputs to an absolute URL with the given prefix override. If failed, return undefined.
 */
const normalizeUrl = (filename: string, prefixOverride?: string) => {
  const baseUrl = prefixOverride ?? scriptSrc;
  try {
    const url = baseUrl ? new URL(filename, baseUrl) : new URL(filename);
    return url.href;
  } catch {
    return undefined;
  }
};

/**
 * Create a fallback URL if an absolute URL cannot be created by the normalizeUrl function.
 */
const fallbackUrl = (filename: string, prefixOverride?: string) => `${prefixOverride ?? './'}${filename}`;

/**
 * This helper function is used to preload a module from a URL.
 *
 * If the origin of the worker URL is different from the current origin, the worker cannot be loaded directly.
 * See discussions in https://github.com/webpack-contrib/worker-loader/issues/154
 *
 * In this case, we will fetch the worker URL and create a new Blob URL with the same origin as a workaround.
 *
 * @param absoluteUrl - The absolute URL to preload.
 *
 * @returns - A promise that resolves to a new Blob URL
 */
const preload = async(absoluteUrl: string): Promise<string> => {
  const response = await fetch(absoluteUrl, {credentials: 'same-origin'});
  const blob = await response.blob();
  return URL.createObjectURL(blob);
};

/**
 * This helper function is used to dynamically import a module from a URL.
 *
 * The build script has special handling for this function to ensure that the URL is not bundled into the final output.
 *
 * @param url - The URL to import.
 *
 * @returns - A promise that resolves to the default export of the module.
 */
const dynamicImportDefault = async<T>(url: string): Promise<T> => (await import(/* webpackIgnore: true */ url)).default;

/**
 * The proxy worker factory imported from the proxy worker module.
 *
 * This is only available when the WebAssembly proxy is not disabled.
 */
const createProxyWorker: ((urlOverride?: string) => Worker)|undefined =
    // eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-var-requires
    BUILD_DEFS.DISABLE_WASM_PROXY ? undefined : require('./proxy-worker/main').default;

/**
 * Import the proxy worker.
 *
 * This function will perform the following steps:
 * 1. If a preload is needed, it will preload the module and return the object URL.
 * 2. Use the proxy worker factory to create the proxy worker.
 *
 * @returns - A promise that resolves to a tuple of 2 elements:
 *            - The object URL of the preloaded module, or undefined if no preload is needed.
 *            - The proxy worker.
 */
export const importProxyWorker = async(): Promise<[undefined | string, Worker]> => {
  if (!scriptSrc) {
    throw new Error('Failed to load proxy worker: cannot determine the script source URL.');
  }

  // If the script source is from the same origin, we can use the embedded proxy module directly.
  if (isSameOrigin(scriptSrc)) {
    return [undefined, createProxyWorker!()];
  }

  // Otherwise, need to preload
  const url = await preload(scriptSrc);
  return [url, createProxyWorker!(url)];
};

/**
 * The embedded WebAssembly module.
 *
 * This is only available in ESM and when embedding is not disabled.
 */
const embeddedWasmModule: EmscriptenModuleFactory<OrtWasmModule>|undefined =
    BUILD_DEFS.IS_ESM && BUILD_DEFS.DISABLE_DYNAMIC_IMPORT ?
    // eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-var-requires
    require(
        !BUILD_DEFS.DISABLE_TRAINING ? '../../dist/ort-training-wasm-simd-threaded.mjs' :
            !BUILD_DEFS.DISABLE_JSEP ? '../../dist/ort-wasm-simd-threaded.jsep.mjs' :
                                       '../../dist/ort-wasm-simd-threaded.mjs')
        .default :
    undefined;

/**
 * Import the WebAssembly module.
 *
 * This function will perform the following steps:
 * 1. If BUILD_DEFS.DISABLE_DYNAMIC_IMPORT is true, use the embedded module.
 * 2. If a preload is needed, it will preload the module and return the object URL.
 * 3. Otherwise, it will perform a dynamic import of the module.
 *
 * @returns - A promise that resolves to a tuple of 2 elements:
 *            - The object URL of the preloaded module, or undefined if no preload is needed.
 *            - The default export of the module, which is a factory function to create the WebAssembly module.
 */
export const importWasmModule = async(
    urlOverride: string|undefined, prefixOverride: string|undefined,
    isMultiThreaded: boolean): Promise<[undefined | string, EmscriptenModuleFactory<OrtWasmModule>]> => {
  if (BUILD_DEFS.DISABLE_DYNAMIC_IMPORT) {
    return [undefined, embeddedWasmModule!];
  } else {
    const wasmModuleFilename = !BUILD_DEFS.DISABLE_TRAINING ? 'ort-training-wasm-simd-threaded.mjs' :
        !BUILD_DEFS.DISABLE_JSEP                            ? 'ort-wasm-simd-threaded.jsep.mjs' :
                                                              'ort-wasm-simd-threaded.mjs';
    const wasmModuleUrl = urlOverride ?? normalizeUrl(wasmModuleFilename, prefixOverride);
    // need to preload if all of the following conditions are met:
    // 1. not in Node.js.
    //    - Node.js does not have the same origin policy for creating workers.
    // 2. multi-threaded is enabled.
    //    - If multi-threaded is disabled, no worker will be created. So we don't need to preload the module.
    // 3. the absolute URL is available.
    //    - If the absolute URL is failed to be created, the origin cannot be determined. In this case, we will not
    //    preload the module.
    // 4. the worker URL is not from the same origin.
    //    - If the worker URL is from the same origin, we can create the worker directly.
    const needPreload = !isNode && isMultiThreaded && wasmModuleUrl && !isSameOrigin(wasmModuleUrl, prefixOverride);
    const url = needPreload ? (await preload(wasmModuleUrl)) :
                              (wasmModuleUrl ?? fallbackUrl(wasmModuleFilename, prefixOverride));
    return [needPreload ? url : undefined, await dynamicImportDefault<EmscriptenModuleFactory<OrtWasmModule>>(url)];
  }
};
