// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const NODEJS_TEST_CASES = [
  './node-test-main-no-threads.js',
  './node-test-main.js',
  './node-test-main.mjs',
  './node-test-wasm-path-override-filename.js',
  './node-test-wasm-path-override-prefix.js',
];

// [test_for_same_origin, test_for_cross_origin, main_js, ort_main_js]
const BROWSER_TEST_CASES = [
  [true, true, './browser-test-webgl.js', 'ort.min.js'],                               // webgl
  [true, true, './browser-test-webgl.js', 'ort.webgl.min.js'],                         // webgl
  [true, true, './browser-test-wasm.js', 'ort.min.js', ['num_threads=2']],             // wasm, 2 threads
  [true, true, './browser-test-wasm.js', 'ort.wasm.min.js'],                           // wasm, ort.wasm
  [true, true, './browser-test-wasm-multi-session-create.js', 'ort.min.js'],           // wasm, multi-session create
  [true, true, './browser-test-wasm.js', 'ort.min.js', ['num_threads=1']],             // wasm, 1 thread
  [true, true, './browser-test-wasm.js', 'ort.wasm-core.min.js'],                      // wasm, ort.wasm-core
  [true, true, './browser-test-wasm.js', 'ort.min.js', ['num_threads=2', 'proxy=1']],  // wasm, 2 threads, proxy
  [true, true, './browser-test-wasm.js', 'ort.min.js', ['num_threads=1', 'proxy=1']],  // wasm, 1 thread, proxy

  // path override:
  // wasm, path override filename, same origin
  [true, false, './browser-test-wasm-path-override-filename.js', 'ort.min.js', ['base=http://localhost:9876/']],
  [true, false, './browser-test-wasm-path-override-filename.js', 'ort.wasm.min.js', ['base=http://localhost:9876/']],
  // wasm, path override filename, cross origin
  [false, true, './browser-test-wasm-path-override-filename.js', 'ort.min.js', ['base=http://localhost:8081/']],
  [false, true, './browser-test-wasm-path-override-filename.js', 'ort.wasm.min.js', ['base=http://localhost:8081/']],
  // wasm, path override prefix, same origin
  [true, false, './browser-test-wasm-path-override-prefix.js', 'ort.min.js', ['base=http://localhost:9876/']],
  [true, false, './browser-test-wasm-path-override-prefix.js', 'ort.wasm.min.js', ['base=http://localhost:9876/']],
  // wasm, path override prefix, cross origin
  [false, true, './browser-test-wasm-path-override-prefix.js', 'ort.min.js', ['base=http://localhost:8081/']],
  [false, true, './browser-test-wasm-path-override-prefix.js', 'ort.wasm.min.js', ['base=http://localhost:8081/']],

  [true, true, './browser-test-wasm-image-tensor-image.js', 'ort.min.js'],      // pre-post-process
  [true, true, './browser-test-webgpu-external-data.js', 'ort.webgpu.min.js'],  // external data
];

const BUNDLER_TEST_CASES = [
  './dist/webpack_esm_js/ort-test-e2e.bundle.mjs',
  './dist/webpack_umd_js/ort-test-e2e.bundle.js',
  './dist/rollup_esm_js/ort-test-e2e.bundle.mjs',
  './dist/rollup_umd_js/ort-test-e2e.bundle.js',
  './dist/parcel_esm_js/main.js',
  './dist/parcel_umd_js/main.js',
];

module.exports = {
  NODEJS_TEST_CASES,
  BROWSER_TEST_CASES,
  BUNDLER_TEST_CASES,
};
