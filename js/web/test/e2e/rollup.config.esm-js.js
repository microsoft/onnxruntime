// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const { nodeResolve } = require('@rollup/plugin-node-resolve');
const terser = require('@rollup/plugin-terser');
const copy = require('rollup-plugin-copy');

module.exports = {
  input: 'src/esm-js/main.js',
  output: {
    file: 'dist/rollup_esm_js/ort-test-e2e.bundle.mjs',
    format: 'esm',
  },
  plugins: [
    // Use "@rollup/plugin-node-resolve" to support conditional import.
    // (e.g. `import {...} from 'onnxruntime-web/wasm';`)
    nodeResolve(),

    // Use "@rollup/plugin-terser" to minify the output.
    terser(),

    // Use "rollup-plugin-copy" to copy the onnxruntime-web WebAssembly files to the output directory.
    copy({ targets: [{ src: 'node_modules/onnxruntime-web/dist/ort-*.{js,mjs,wasm}', dest: 'dist/rollup_esm_js' }] }),
  ],
};
