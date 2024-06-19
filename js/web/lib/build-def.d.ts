// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* eslint-disable @typescript-eslint/naming-convention */

/**
 * The interface BuildDefinitions contains a set of flags which are defined at build time.
 *
 * Those flags are processed in bundler for tree shaking to remove unused code.
 * No flags in this file should present in production build.
 */
interface BuildDefinitions {
  // #region Build definitions for Tree Shaking

  /**
   * defines whether to disable the whole WebGL backend in the build.
   */
  readonly DISABLE_WEBGL: boolean;
  /**
   * defines whether to disable the whole WebGpu/WebNN backend in the build.
   */
  readonly DISABLE_JSEP: boolean;
  /**
   * defines whether to disable the whole WebNN backend in the build.
   */
  readonly DISABLE_WASM: boolean;
  /**
   * defines whether to disable proxy feature in WebAssembly backend in the build.
   */
  readonly DISABLE_WASM_PROXY: boolean;
  /**
   * defines whether to disable training APIs in WebAssembly backend.
   */
  readonly DISABLE_TRAINING: boolean;
  /**
   * defines whether to disable dynamic importing WASM module in the build.
   */
  readonly DISABLE_DYNAMIC_IMPORT: boolean;

  // #endregion

  // #region Build definitions for ESM

  /**
   * defines whether the build is ESM.
   */
  readonly IS_ESM: boolean;
  /**
   * placeholder for the import.meta.url in ESM. in CJS, this is undefined.
   */
  readonly ESM_IMPORT_META_URL: string|undefined;

  // #endregion
}

declare const BUILD_DEFS: BuildDefinitions;
