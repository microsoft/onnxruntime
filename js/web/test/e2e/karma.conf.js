// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const args = require('minimist')(process.argv.slice(2));
const SELF_HOST = !!args['self-host'];
const ORT_MAIN = args['ort-main'];
const TEST_MAIN = args['test-main'];
if (typeof TEST_MAIN !== 'string') {
  throw new Error('flag --test-main=<TEST_MAIN_JS_FILE> is required');
}
const USER_DATA = args['user-data'];
if (typeof USER_DATA !== 'string') {
  throw new Error('flag --user-data=<CHROME_USER_DATA_FOLDER> is required');
}

const FORMAT = args['format'];
if (FORMAT !== 'esm' && FORMAT !== 'iife') {
  throw new Error('flag --format=<esm|iife> is required');
}

const ENABLE_SHARED_ARRAY_BUFFER = !!args['enable-shared-array-buffer'];

const testArgs = args['test-args'];
const normalizedTestArgs = !testArgs || Array.isArray(testArgs) ? testArgs : [testArgs];

const files = [
  { pattern: './model.onnx', included: false },
  { pattern: './model_with_orig_ext_data.onnx', included: false },
  { pattern: './model_with_orig_ext_data.bin', included: false },
  { pattern: './test-wasm-path-override/*', included: false, nocache: true, watched: false },
];
if (ORT_MAIN) {
  if (ORT_MAIN.endsWith('.mjs')) {
    files.push({
      pattern: (SELF_HOST ? './esm-loaders/' : 'http://localhost:8081/esm-loaders/') + ORT_MAIN,
      type: 'module',
    });
  } else {
    files.push({
      pattern: (SELF_HOST ? './node_modules/onnxruntime-web/dist/' : 'http://localhost:8081/dist/') + ORT_MAIN,
    });
  }
}
if (FORMAT === 'esm') {
  files.push({ pattern: TEST_MAIN, type: 'module' });
} else {
  files.push({ pattern: './common.js' }, { pattern: TEST_MAIN });
}
files.push({ pattern: './dist/**/*', included: false, nocache: true, watched: false });
if (SELF_HOST) {
  files.push({ pattern: './node_modules/onnxruntime-web/dist/*.*', included: false, nocache: true });
}

const flags = ['--ignore-gpu-blocklist', '--gpu-vendor-id=0x10de'];
if (ENABLE_SHARED_ARRAY_BUFFER) {
  flags.push('--enable-features=SharedArrayBuffer');
}

module.exports = function (config) {
  config.set({
    frameworks: ['mocha'],
    files,
    plugins: [require('@chiragrupani/karma-chromium-edge-launcher'), ...config.plugins],
    proxies: {
      '/model.onnx': '/base/model.onnx',
      '/model_with_orig_ext_data.onnx': '/base/model_with_orig_ext_data.onnx',
      '/model_with_orig_ext_data.bin': '/base/model_with_orig_ext_data.bin',
      '/test-wasm-path-override/': '/base/test-wasm-path-override/',
    },
    client: { captureConsole: true, args: normalizedTestArgs, mocha: { expose: ['body'], timeout: 60000 } },
    reporters: ['mocha'],
    captureTimeout: 120000,
    reportSlowerThan: 100,
    browserDisconnectTimeout: 600000,
    browserNoActivityTimeout: 300000,
    browserDisconnectTolerance: 0,
    browserSocketTimeout: 60000,
    hostname: 'localhost',
    browsers: [],
    customLaunchers: {
      Chrome_default: { base: 'Chrome', flags, chromeDataDir: USER_DATA },
      Chrome_no_threads: {
        base: 'Chrome',
        chromeDataDir: USER_DATA,
        flags,
        // TODO: no-thread flags
      },
      Edge_default: { base: 'Edge', edgeDataDir: USER_DATA },
    },
  });
};
