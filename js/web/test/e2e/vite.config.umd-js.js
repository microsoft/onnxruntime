// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const path = require('node:path');
const { defineConfig } = require('vite');

module.exports = defineConfig({
  build: {
    outDir: path.resolve(__dirname, 'dist/vite_umd_js'),
    emptyOutDir: true,
    sourcemap: false,
    lib: {
      name: 'testPackageConsuming',
      entry: path.resolve(__dirname, 'src/esm-js/vite-main.js'),
      fileName: () => 'ort-test-e2e.bundle.js',
      formats: ['umd'],
    },
    minify: false,
    assetsDir: './',
    assetsInlineLimit: 0,
    commonjsOptions: {
      include: ['**/*.js'],
      exclude: [],
      transformMixedEsModules: true,
      ignoreDynamicRequires: true,
    },
  },
});
