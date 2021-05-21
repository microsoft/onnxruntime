// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const path = require('path');
const webpack = require('webpack');
const NodePolyfillPlugin = require('node-polyfill-webpack-plugin');
const TerserPlugin = require("terser-webpack-plugin");
const minimist = require('minimist');

const VERSION = require(path.join(__dirname, 'package.json')).version;
const COPYRIGHT_BANNER = `/*!
* ONNX Runtime Web v${VERSION}
* Copyright (c) Microsoft Corporation. All rights reserved.
* Licensed under the MIT License.
*/`;

function defaultTerserPluginOptions() {
  return {
    extractComments: false,
    terserOptions: {
      format: {
        comments: false,
      },
      compress: {
        passes: 2
      },
      mangle: {
        reserved: ["_scriptDir"]
      }
    }
  };
}

// common config for release bundle
function buildConfig({ filename, format, target, mode, devtool }) {
  const config = {
    target: [format === 'commonjs' ? 'node' : 'web', target],
    entry: path.resolve(__dirname, 'lib/index.ts'),
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename,
      library: {
        type: format
      }
    },
    resolve: {
      extensions: ['.ts', '.js'],
      alias: {
        "util": false,
      },
      fallback: {
        "fs": false,
        "path": false,
        "util": false,
        "os": false,
        "worker_threads": false,
        "perf_hooks": false,
      }
    },
    plugins: [new webpack.WatchIgnorePlugin({ paths: [/\.js$/, /\.d\.ts$/] })],
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
      }, {
        test: /\.worker.js$/,
        type: 'asset/source'
      }]
    },
    mode,
    devtool
  };

  if (mode === 'production') {
    config.resolve.alias['./binding/ort-wasm-threaded.js'] = './binding/ort-wasm-threaded.min.js';
    config.resolve.alias['./binding/ort-wasm-threaded.worker.js'] = './binding/ort-wasm-threaded.min.worker.js';

    const options = defaultTerserPluginOptions();
    options.terserOptions.format.preamble = COPYRIGHT_BANNER;
    config.plugins.push(new TerserPlugin(options));
  } else {
    config.plugins.push(new webpack.BannerPlugin({ banner: COPYRIGHT_BANNER, raw: true }));
  }

  return config;
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
    config.externals.worker_threads = 'worker_threads';
    config.externals.perf_hooks = 'perf_hooks';
    config.externals.os = 'os';
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
  const config = {
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
      '../../node': '../../node'
    },
    resolve: {
      extensions: ['.ts', '.js'],
      aliasFields: [],
      fallback: {
        './binding/ort-wasm.js': false,
        './binding/ort-wasm-threaded.js': false,
        './binding/ort-wasm-threaded.worker.js': false
      }
    },
    plugins: [
      new webpack.WatchIgnorePlugin({ paths: [/\.js$/, /\.d\.ts$/] }),
      new NodePolyfillPlugin(),
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
      }, {
        test: /\.worker\.js$/,
        type: 'asset/source'
      }]
    },
    mode: mode,
    devtool: devtool,
  };

  if (mode === 'production') {
    config.plugins.push(new TerserPlugin(defaultTerserPluginOptions()));
  }

  return config;
}

module.exports = () => {
  const args = minimist(process.argv);
  const bundleMode = args['bundle-mode'] || 'prod';  // 'prod'|'dev'|'perf'|'node'|undefined;
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
      );

    case 'node':
      builds.push(
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
