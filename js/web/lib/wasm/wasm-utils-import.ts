// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
 * This helper function is used to preload a module from a URL.
 *
 * If the origin of the module URL is different from the current origin, it will fetch the worker URL and create a new
 * Blob URL with the same origin.
 *
 * @param filename - The file name of the module to preload.
 * @param prefixOverride - an optional prefixOverride.
 * @param noPreload - an optional flag to skip preloading the module.
 *
 * @returns - A promise that resolves to a tuple containing 2 elements:
 *            - A boolean indicating whether the module is preloaded.
 *            - The URL of the module to import. If the module is preloaded, it will use a new Blob URL, otherwise the
 *              normalized URL.
 */
const preload =
    async(filename: string, prefixOverride?: string, noPreload?: boolean): Promise<readonly[boolean, string]> => {
  const fallback = [false, `${prefixOverride ?? './'}${filename}`] as const;
  if (isNode || noPreload) {
    // skip preload worker in Node.js or when noPreload is set explicitly
    return fallback;
  }

  const baseUrl = prefixOverride ?? scriptSrc;
  let url: URL;
  try {
    url = baseUrl ? new URL(filename, baseUrl) : new URL(filename);
  } catch {
    // if failed to resolve from workerUrl and baseUrl, we should skip and return the original workerUrl
    return fallback;
  }

  if (origin && url.origin !== origin) {
    // if origin is different, preload worker

    try {
      // because the origin is different, we need to create a new Blob URL with the same origin
      const response = await fetch(url);
      const blob = await response.blob();
      return [true, URL.createObjectURL(blob)];
    } catch (e) {
      // if failed to createURL from prefixOverride, it is not a valid URL, so we should ignore it
      // eslint-disable-next-line no-console
      console.warn(`Failed to preload worker from FileName="${filename}", Base="${baseUrl ?? ''}": ${e}`);
    }
  }

  return [false, url.href];
};

/**
 * This helper function is used to dynamically import a module from a URL.
 *
 * This function is used in 2 places:
 * - For the proxy feature, it is used to dynamically import the proxy module.
 * - For WebAssembly, it is used to dynamically import the WebAssembly module.
 *
 * The 2 modules are similar in the following ways:
 * - Both modules are standalone ESM format.
 * - Both are not bundled into the final output.
 * - Both will load themselves as an entry point of a worker.
 *
 * If the origin of the worker URL is different from the current origin, the worker cannot be loaded directly.
 * See discussions in https://github.com/webpack-contrib/worker-loader/issues/154
 *
 * In this case, we will fetch the worker URL and create a new Blob URL with the same origin as a workaround.
 *
 * The build script has special handling for this function to ensure that the URL is not bundled into the final output.
 *
 * @param filename - The file name of the module to import.
 * @param prefixOverride - an optional config for prefix override.
 * @param noPreload - an boolean value indicating whether to skip preloading the module. Specifically, this flag should
 *     be set to true when loading wasm module with numThreads === 1, because no worker is needed.
 *
 * @returns - A promise that resolves to a tuple containing 2 elements:
 *            - A string representing the URL, if it is created by `URL.createObjectURL()`; otherwise, undefined.
 *            - The default export of the module.
 */
export const dynamicImportDefault =
    async<T>(filename: string, prefixOverride?: string, noPreload = false): Promise<[undefined | string, T]> => {
  const [preloaded, url] = await preload(filename, prefixOverride, noPreload);
  return [preloaded ? url : undefined, (await import(/* webpackIgnore: true */ url)).default];
};
