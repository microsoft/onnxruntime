// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const path = require('path');
const webpack = require('webpack');
const TerserPlugin = require("terser-webpack-plugin");

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

function addCopyrightBannerPlugin(mode, target) {
  const VERSION = require(path.join(__dirname, 'package.json')).version;
  const COPYRIGHT_BANNER = `/*!
 * ONNX Runtime Common v${VERSION}
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */`;

  if (mode === 'production') {
    return new TerserPlugin({
      extractComments: false,
      terserOptions: {
        ecma: terserEcmaVersionFromWebpackTarget(target),
        format: {
          preamble: COPYRIGHT_BANNER,
          comments: false,
        },
        compress: {
          passes: 2
        }
      }
    });
  } else {
    return new webpack.BannerPlugin({ banner: COPYRIGHT_BANNER, raw: true });
  }
}

function buildConfig({
  suffix = '',
  format = 'umd',
  target = 'es2017',
  mode = 'production',
  devtool = 'source-map'
}) {
  return {
    target: [format === 'commonjs' ? 'node' : 'web', target],
    entry: path.resolve(__dirname, 'lib/index.ts'),
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: `ort-common${suffix}.js`,
      library: {
        name: format === 'commonjs' ? undefined : 'ort',
        type: format
      }
    },
    resolve: { extensions: ['.ts', '.js'] },
    plugins: [
      new webpack.WatchIgnorePlugin({ paths: [/\.js$/, /\.d\.ts$/] }),
      addCopyrightBannerPlugin(mode, target),
    ],
    module: {
      rules: [{
        test: /\.tsx?$/,
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
    mode: mode,
    devtool: devtool,
  };
}

module.exports = (env, argv) => {
  return [
    buildConfig({ suffix: '.es5.min', target: 'es5' }),
    buildConfig({ suffix: '.es6.min', target: 'es6' }),
    buildConfig({ suffix: '.min' }),
    buildConfig({ mode: 'development', devtool: 'inline-source-map' }),
    buildConfig({ format: 'commonjs', suffix: '.node' }),
  ];
};
