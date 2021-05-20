// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const args = require('minimist')(process.argv.slice(2));
const SELF_HOST = !!args['self-host'];
const TEST_MAIN = args['test-main'];
if (typeof TEST_MAIN !== 'string') {
  throw new Error('flag --test-main=<TEST_MAIN_JS_FILE> is required');
}
const USER_DATA = args['user-data'];
if (typeof USER_DATA !== 'string') {
  throw new Error('flag --user-data=<CHROME_USER_DATA_FOLDER> is required');
}

module.exports = function (config) {
  const distPrefix = SELF_HOST ? './node_modules/onnxruntime-web/dist/' : 'http://localhost:8081/dist/';
  config.set({
    frameworks: ['mocha'],
    files: [
      { pattern: distPrefix + 'ort.js' },
      { pattern: './common.js' },
      { pattern: TEST_MAIN },
      { pattern: './node_modules/onnxruntime-web/dist/**/*', included: false, nocache: true },
      { pattern: './model.onnx', included: false }
    ],
    proxies: {
      '/model.onnx': '/base/model.onnx',
    },
    client: { captureConsole: true, mocha: { expose: ['body'], timeout: 60000 } },
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
      Chrome_default: {
        base: 'Chrome',
        chromeDataDir: USER_DATA
      },
      Chrome_no_threads: {
        base: 'Chrome',
        chromeDataDir: USER_DATA,
        // TODO: no-thread flags
      }
    }
  });
};
