// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const path = require('node:path');
const CopyPlugin = require('copy-webpack-plugin');

module.exports = {
  module: { parser: { javascript: { importMeta: false } } },
  experiments: { outputModule: true },
  target: ['web'],
  entry: path.resolve(__dirname, 'src/esm-js/main.js'),
  output: {
    clean: true,
    filename: 'ort-test-e2e.bundle.mjs',
    path: path.resolve(__dirname, 'dist/webpack_esm_js'),
    library: { type: 'module' },
  },
  plugins: [
    // Use "copy-webpack-plugin" to copy the onnxruntime-web WebAssembly files to the output directory.
    new CopyPlugin({
      patterns: [{ from: 'node_modules/onnxruntime-web/dist/ort-*.{js,mjs,wasm}', to: '[name][ext]' }],
    }),
  ],
};
