// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const path = require('path');
const webpack = require('webpack');
const BundleAnalyzerPlugin = require('webpack-bundle-analyzer').BundleAnalyzerPlugin;
const NodePolyfillPlugin = require('node-polyfill-webpack-plugin');
const TerserPlugin = require("terser-webpack-plugin");
const minimist = require('minimist');

// commandline args
const args = minimist(process.argv);
const bundleMode = args['bundle-mode'] || 'prod';  // 'prod'|'dev'|'perf'|'node'|undefined;
const useAnalyzer = !!args.a || !!args['use-analyzer'];  // -a, --use-analyzer
const filter = args.f || args.filter;

const VERSION = require(path.join(__dirname, 'package.json')).version;
const COPYRIGHT_BANNER = `/*!
* ONNX Runtime Web v${VERSION}
* Copyright (c) Microsoft Corporation. All rights reserved.
* Licensed under the MIT License.
*/`;

function terserEcmaVersionFromWebpackTarget(target) {
  switch (target) {
    case 'es5':
      return 5;
    case 'es6':
    case 'es2015':
      return 2015;
    case 'es2017':
      return 2017;
    default:
      throw new RangeError(`not supported ECMA version: ${target}`);
  }
}

function defaultTerserPluginOptions(target) {
  return {
    extractComments: false,
    terserOptions: {
      ecma: terserEcmaVersionFromWebpackTarget(target),
      format: {
        comments: false,
      },
      compress: {
        passes: 2
      },
      mangle: {
        reserved: ["_scriptDir", "startWorker"]
      }
    }
  };
}

const DEFAULT_BUILD_DEFS = {
  DISABLE_WEBGL: false,
  DISABLE_WEBGPU: true,
  DISABLE_WASM: false,
  DISABLE_WASM_PROXY: false,
  DISABLE_WASM_THREAD: false,
};

// common config for release bundle
function buildConfig({ filename, format, target, mode, devtool, build_defs }) {
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
        "crypto": false,
        "fs": false,
        "path": false,
        "util": false,
        "os": false,
        "worker_threads": false,
        "perf_hooks": false,
      }
    },
    plugins: [
      new webpack.DefinePlugin({ BUILD_DEFS: build_defs }),
      new webpack.WatchIgnorePlugin({ paths: [/\.js$/, /\.d\.ts$/] })
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
      }, {
        test: /ort-wasm.*\.worker\.js$/,
        type: 'asset/source'
      }]
    },
    mode,
    node: false,
    devtool
  };

  if (useAnalyzer) {
    config.plugins.unshift(new BundleAnalyzerPlugin({
      analyzerMode: 'static',
      reportFilename: `${filename}.report.html`
    }));
  }

  if (mode === 'production') {
    config.resolve.alias['./binding/ort-wasm-threaded.js'] = './binding/ort-wasm-threaded.min.js';
    config.resolve.alias['./binding/ort-wasm-threaded-simd.jsep.js'] = './binding/ort-wasm-threaded-simd.jsep.min.js';
    config.resolve.alias['./binding/ort-wasm-threaded.worker.js'] = './binding/ort-wasm-threaded.min.worker.js';

    const options = defaultTerserPluginOptions(target);
    options.terserOptions.format.preamble = COPYRIGHT_BANNER;
    config.plugins.push(new TerserPlugin(options));

    // add a custom plugin to check whether code contains 'BUILD_DEFS'
    config.plugins.push({
      apply: (compiler) => {
        compiler.hooks.afterCompile.tap(
          'Check BUILD_DEFS',
          (compilation) => {
            for (const filename of compilation.assetsInfo.keys()) {
              if (filename.endsWith('.js')) {
                const asset = compilation.assets[filename];
                if (asset) {
                  const content = asset.source();
                  if (typeof content !== 'string') {
                    throw new Error(`content for target file '${filename}' is not string.`);
                  }
                  if (content.includes('DISABLE_WEBGL')
                    || content.includes('DISABLE_WASM')
                    || content.includes('DISABLE_WASM_PROXY')
                    || content.includes('DISABLE_WASM_THREAD')) {
                    throw new Error(`target file '${filename}' contains data fields from "BUILD_DEFS".`);
                  }
                }
              }
            }
          });
      }
    });
  } else {
    config.plugins.push(new webpack.BannerPlugin({ banner: COPYRIGHT_BANNER, raw: true }));
  }

  return config;
}

// "ort{.min}.js" config
function buildOrtConfig({
  suffix = '',
  target = 'es2017',
  mode = 'production',
  devtool = 'source-map',
  build_defs = DEFAULT_BUILD_DEFS
}) {
  const config = buildConfig({ filename: `ort${suffix}.js`, format: 'umd', target, mode, devtool, build_defs });
  // set global name 'ort'
  config.output.library.name = 'ort';
  return config;
}

// "ort-web{.min|.node}.js" config
function buildOrtWebConfig({
  suffix = '',
  format = 'umd',
  target = 'es2017',
  mode = 'production',
  devtool = 'source-map',
  build_defs = DEFAULT_BUILD_DEFS
}) {
  const config = buildConfig({ filename: `ort-web${suffix}.js`, format, target, mode, devtool, build_defs });
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
  target = 'es2017',
  mode = 'production',
  devtool = 'source-map',
  build_defs = DEFAULT_BUILD_DEFS
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
      alias: {
        // make sure to refer to original source files instead of generated bundle in test-main.
        '..$': '../lib/index'
      },
      extensions: ['.ts', '.js'],
      fallback: {
        './binding/ort-wasm.js': false,
        './binding/ort-wasm-threaded.js': false,
        './binding/ort-wasm-threaded.worker.js': false
      }
    },
    plugins: [
      new webpack.DefinePlugin({ BUILD_DEFS: build_defs }),
      new webpack.WatchIgnorePlugin({ paths: [/\.js$/, /\.d\.ts$/] }),
      new NodePolyfillPlugin({
        excludeAliases: ["console", "Buffer"]
      }),
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
      }, {
        test: /ort-wasm.*\.worker\.js$/,
        type: 'asset/source'
      }]
    },
    mode,
    node: false,
    devtool,
  };

  if (mode === 'production') {
    config.plugins.push(new TerserPlugin(defaultTerserPluginOptions(target)));
  }

  return config;
}

module.exports = () => {
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
        // ort.es5.min.js
        buildOrtConfig({ suffix: '.es5.min', target: 'es5' }),

        // ort.wasm.min.js
        buildOrtConfig({
          suffix: '.wasm.min', build_defs: {
            ...DEFAULT_BUILD_DEFS,
            DISABLE_WEBGL: true,
          }
        }),
        // ort.webgl.min.js
        buildOrtConfig({
          suffix: '.webgl.min', build_defs: {
            ...DEFAULT_BUILD_DEFS,
            DISABLE_WASM: true,
          }
        }),
        // ort.wasm-core.min.js
        buildOrtConfig({
          suffix: '.wasm-core.min', build_defs: {
            ...DEFAULT_BUILD_DEFS,
            DISABLE_WEBGL: true,
            DISABLE_WASM_PROXY: true,
            DISABLE_WASM_THREAD: true,
          }
        }),
        // ort.webgpu.min.js
        buildOrtConfig({
          suffix: '.webgpu.min', build_defs: {
            ...DEFAULT_BUILD_DEFS,
            DISABLE_WEBGPU: false,
          }
        }),

        // ort-web.min.js
        buildOrtWebConfig({ suffix: '.min' }),
        // ort-web.js
        buildOrtWebConfig({ mode: 'development', devtool: 'inline-source-map' }),
        // ort-web.es6.min.js
        buildOrtWebConfig({ suffix: '.es6.min', target: 'es6' }),
        // ort-web.es5.min.js
        buildOrtWebConfig({ suffix: '.es5.min', target: 'es5' }),
      );

    case 'node':
      builds.push(
        // ort-web.node.js
        buildOrtWebConfig({ suffix: '.node', format: 'commonjs' }),
      );
      break;
    case 'dev':
      builds.push(buildTestRunnerConfig({
        suffix: '.dev', mode: 'development', devtool: 'inline-source-map', build_defs: {
          ...DEFAULT_BUILD_DEFS,
          DISABLE_WEBGPU: false,
        }
      }));
      break;
    case 'perf':
      builds.push(buildTestRunnerConfig({
        suffix: '.perf', build_defs: {
          ...DEFAULT_BUILD_DEFS,
          DISABLE_WEBGPU: false,
        }
      }));
      break;
    default:
      throw new Error(`unsupported bundle mode: ${bundleMode}`);
  }

  if (filter) {
    const filterRegex = new RegExp(filter);
    return builds.filter(b => filterRegex.test(b.output.filename));
  }

  return builds;
};
