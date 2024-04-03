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
 * This helper function is used to dynamically import a module from a URL.
 *
 * The build script has special handling for this function to ensure that the URL is not bundled into the final output.
 */
export const dynamicImportDefault = async (filename: string, prefixOverride?: string) =>
    (await import(/* webpackIgnore: true */ `${prefixOverride ?? './'}${filename}`)).default;

/**
 * This helper function is used to preload a worker from a URL.
 *
 * If the origin of the worker URL is different from the current origin, the worker cannot be loaded directly.
 * See discussions in https://github.com/webpack-contrib/worker-loader/issues/154
 *
 * In this case, we will fetch the worker URL and create a new Blob URL with the same origin as a workaround.
 *
 * @param filename - The file name of the worker to preload.
 * @param baseUrl - an optional base URL to resolve the filename.
 * @returns - The URL of the worker to use. If the worker is preloaded, it will return a new Blob URL, otherwise the
 *     original URL.
 */
export const preloadWorker = async(filename: string, baseUrl?: string): Promise<string> => {
  if (isNode) {
    // skip preload worker in Node.js
    return filename;
  }

  let url: URL;
  try {
    url = baseUrl ? new URL(filename, baseUrl) : new URL(filename);
  } catch {
    // if failed to resolve from workerUrl and baseUrl, we should skip and return the original workerUrl
    return filename;
  }

  if (origin && url.origin !== origin) {
    // if origin is different, preload worker

    try {
      // because the origin is different, we need to create a new Blob URL with the same origin
      const response = await fetch(url);
      const blob = await response.blob();

      return URL.createObjectURL(blob);
    } catch (e) {
      // if failed to createURL from prefixOverride, it is not a valid URL, so we should ignore it
      // eslint-disable-next-line no-console
      console.warn(`Failed to preload worker from FileName="${filename}", Base="${baseUrl ?? ''}": ${e}`);
    }
  }

  return url.href;
};
