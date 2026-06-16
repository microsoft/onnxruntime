// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const path = require('node:path');
const { defineConfig } = require('vite');

module.exports = defineConfig({
  build: {
    outDir: path.resolve(__dirname, 'dist/vite_esm_js'),
    emptyOutDir: true,
    sourcemap: false,
    lib: {
      name: 'testPackageConsuming',
      entry: path.resolve(__dirname, 'src/esm-js/vite-main.js'),
      fileName: () => 'ort-test-e2e.bundle.mjs',
      formats: ['es'],
    },
    minify: false,
    assetsDir: './',
    assetsInlineLimit: 0,
  },
});
