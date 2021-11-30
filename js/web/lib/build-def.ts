// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* eslint-disable @typescript-eslint/no-unused-vars, @typescript-eslint/naming-convention */

/**
 * The interface BuildDefinitions contains a set of flags which are defined at build time.
 *
 * Those flags are processed in terser for tree shaking to remove unused code.
 * No flags in this file should present in production build.
 */
interface BuildDefinitions {
  /**
   * defines whether to disable the whole WebGL backend in the build.
   */
  DISABLE_WEBGL: boolean;
  /**
   * defines whether to disable the whole WebAssembly backend in the build.
   */
  DISABLE_WASM: boolean;
  /**
   * defines whether to disable proxy feature in WebAssembly backend in the build.
   */
  DISABLE_WASM_PROXY: boolean;
  /**
   * defines whether to disable multi-threading feature in WebAssembly backend in the build.
   */
  DISABLE_WASM_THREAD: boolean;
}

declare let BUILD_DEFS: BuildDefinitions;
