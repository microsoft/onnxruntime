// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const commonjs = require('@rollup/plugin-commonjs');
const { nodeResolve } = require('@rollup/plugin-node-resolve');
const terser = require('@rollup/plugin-terser');
const copy = require('rollup-plugin-copy');

module.exports = {
  input: 'src/cjs-js/main.js',
  output: {
    name: 'testPackageConsuming',
    file: 'dist/rollup_umd_js/ort-test-e2e.bundle.js',
    format: 'umd',
  },
  plugins: [
    // Use "@rollup/plugin-node-resolve" to support conditional import.
    // (e.g. `import {...} from 'onnxruntime-web/wasm';`)
    nodeResolve(),

    // Use "@rollup/plugin-commonjs" to support CommonJS module resolve.
    commonjs({ ignoreDynamicRequires: true }),

    // Use "@rollup/plugin-terser" to minify the output.
    terser(),

    // Use "rollup-plugin-copy" to copy the onnxruntime-web WebAssembly files to the output directory.
    copy({ targets: [{ src: 'node_modules/onnxruntime-web/dist/ort-*.{js,mjs,wasm}', dest: 'dist/rollup_umd_js' }] }),
  ],
};
