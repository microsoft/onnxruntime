// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const path = require('path');
const webpack = require('webpack');
const NodePolyfillPlugin = require("node-polyfill-webpack-plugin");
const minimist = require('minimist');

// common config for release bundle
function buildConfig({ filename, format, target, mode, devtool }) {
  return {
    entry: path.resolve(__dirname, 'lib/index.ts'),
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename,
      library: {
        type: format
      }
    },
    resolve: { extensions: ['.ts', '.js'] },
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
    },
    resolve: { extensions: ['.ts', '.js'], aliasFields: [] },
    plugins: [
      new webpack.WatchIgnorePlugin({ paths: [/\.js$/, /\.d\.ts$/] }),
      new NodePolyfillPlugin()
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

  if (bundleMode === 'prod') {
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
      // ort-web.min.js
      buildOrtWebConfig({ suffix: '.es6.min', target: 'es6' }),
      // ort-web.js
      buildOrtWebConfig({ suffix: '.es6', mode: 'development', devtool: 'inline-source-map', target: 'es6' }),

      // ort-web.node.js
      buildOrtWebConfig({ suffix: '.node', format: 'commonjs' }),
    );
  }

  if (bundleMode === 'dev') {
    builds.push(buildTestRunnerConfig({ suffix: '.dev', mode: 'development', devtool: 'inline-source-map' }));
  } else if (bundleMode === 'perf') {
    builds.push(buildTestRunnerConfig({ suffix: '.perf', devtool: undefined }));
  }

  return builds;
};
