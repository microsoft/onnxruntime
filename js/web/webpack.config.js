// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const path = require('path');
const webpack = require('webpack');
const NodePolyfillPlugin = require('node-polyfill-webpack-plugin');
const TerserPlugin = require("terser-webpack-plugin");
const minimist = require('minimist');

function addCopyrightBannerPlugin(mode) {
  const VERSION = require(path.join(__dirname, 'package.json')).version;
  const COPYRIGHT_BANNER = `/*!
 * ONNX Runtime Web v${VERSION}
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */`;

  if (mode === 'production') {
    return new TerserPlugin({
      extractComments: false,
      terserOptions: {
        format: {
          preamble: COPYRIGHT_BANNER,
          comments: false,
        },
        compress: {
          passes: 2
        },
        mangle: {
          reserved: ["_scriptDir"]
        }
      }
    });
  } else {
    return new webpack.BannerPlugin({ banner: COPYRIGHT_BANNER, raw: true });
  }
}

// common config for release bundle
function buildConfig({ filename, format, target, mode, devtool }) {
  return {
    target: [format === 'commonjs' ? 'node' : 'web', target],
    entry: path.resolve(__dirname, 'lib/index.ts'),
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename,
      library: {
        type: format
      }
    },
    resolve: { extensions: ['.ts', '.js'] },
    plugins: [
      new webpack.WatchIgnorePlugin({ paths: [/\.js$/, /\.d\.ts$/] }),
      addCopyrightBannerPlugin(mode),
    ],
    module: {
      rules: [{
        test: /\.ts$/,
        use: [
          {
            loader: 'ts-loader',
            options: {
              compilerOptions: { target }
            }
          }
        ]
      }]
    },
    mode,
    devtool
  };
}

// "ort{.min}.js" config
function buildOrtConfig({
  suffix = '',
  target = 'es5',
  mode = 'production',
  devtool = 'source-map'
}) {
  const config = buildConfig({ filename: `ort${suffix}.js`, format: 'umd', target, mode, devtool });
  // set global name 'ort'
  config.output.library.name = 'ort';
  // do not use those node builtin modules in browser
  config.resolve.fallback = { path: false, fs: false, util: false };
  return config;
}

// "ort-web{.min|.node}.js" config
function buildOrtWebConfig({
  suffix = '',
  format = 'umd',
  target = 'es5',
  mode = 'production',
  devtool = 'source-map'
}) {
  const config = buildConfig({ filename: `ort-web${suffix}.js`, format, target, mode, devtool });
  // exclude onnxruntime-common from bundle
  config.externals = {
    'onnxruntime-common': {
      commonjs: "onnxruntime-common",
      commonjs2: "onnxruntime-common",
      root: 'ort'
    }
  };
  // in nodejs, treat as external dependencies
  if (format === 'commonjs') {
    config.externals.path = 'path';
    config.externals.fs = 'fs';
    config.externals.util = 'util';
  }
  // in browser, do not use those node builtin modules
  if (format === 'umd') {
    config.resolve.fallback = { path: false, fs: false, util: false };
  }
  return config;
}

function buildTestRunnerConfig({
  suffix = '',
  format = 'umd',
  target = 'es5',
  mode = 'production',
  devtool = 'source-map'
}) {
  return {
    target: ['web', target],
    entry: path.resolve(__dirname, 'test/test-main.ts'),
    output: {
      path: path.resolve(__dirname, 'test'),
      filename: `ort${suffix}.js`,
      library: {
        type: format
      },
      devtoolNamespace: '',
    },
    externals: {
      'onnxruntime-common': 'ort',
      'fs': 'fs',
      'perf_hooks': 'perf_hooks',
      'worker_threads': 'worker_threads',
    },
    resolve: {
      extensions: ['.ts', '.js'],
      aliasFields: [],
      fallback: { './binding/ort-wasm-threaded.js': false, './binding/ort-wasm.js': false }
    },
    plugins: [
      new webpack.WatchIgnorePlugin({ paths: [/\.js$/, /\.d\.ts$/] }),
      new NodePolyfillPlugin(),
      addCopyrightBannerPlugin(mode),
    ],
    module: {
      rules: [{
        test: /\.ts$/,
        use: [
          {
            loader: 'ts-loader',
            options: {
              compilerOptions: { target: target }
            }
          }
        ]
      }]
    },
    mode: mode,
    devtool: devtool,
  };
}

module.exports = () => {
  const args = minimist(process.argv);
  const bundleMode = args['bundle-mode'] || 'prod';  // 'prod'|'dev'|'perf'|undefined;
  const builds = [];

  switch (bundleMode) {
    case 'prod':
      builds.push(
        // ort.min.js
        buildOrtConfig({ suffix: '.min' }),
        // ort.js
        buildOrtConfig({ mode: 'development', devtool: 'inline-source-map' }),
        // ort.es6.min.js
        buildOrtConfig({ suffix: '.es6.min', target: 'es6' }),
        // ort.es6.js
        buildOrtConfig({ suffix: '.es6', mode: 'development', devtool: 'inline-source-map', target: 'es6' }),

        // ort-web.min.js
        buildOrtWebConfig({ suffix: '.min' }),
        // ort-web.js
        buildOrtWebConfig({ mode: 'development', devtool: 'inline-source-map' }),
        // ort-web.es6.min.js
        buildOrtWebConfig({ suffix: '.es6.min', target: 'es6' }),
        // ort-web.es6.js
        buildOrtWebConfig({ suffix: '.es6', mode: 'development', devtool: 'inline-source-map', target: 'es6' }),

        // ort-web.node.js
        buildOrtWebConfig({ suffix: '.node', format: 'commonjs' }),
      );
      break;
    case 'dev':
      builds.push(buildTestRunnerConfig({ suffix: '.dev', mode: 'development', devtool: 'inline-source-map' }));
      break;
    case 'perf':
      builds.push(buildTestRunnerConfig({ suffix: '.perf' }));
      break;
    default:
      throw new Error(`unsupported bundle mode: ${bundleMode}`);
  }

  return builds;
};
