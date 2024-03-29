// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const {nodeResolve} = require('@rollup/plugin-node-resolve');
const copy = require('rollup-plugin-copy');

module.exports = {
  input : 'src/esm-js/main.js',
  output : {
    file : 'dist/rollup_esm_js/ort-test-e2e.bundle.mjs',
    format : 'esm',
    sourcemap : true,
  },
  plugins :
  [
    nodeResolve(),
    copy({targets : [{src : 'node_modules/onnxruntime-web/dist/ort-*.{js,wasm}', dest : 'dist/rollup_esm_js'}]})
  ]
};
