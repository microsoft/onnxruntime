const path = require('path');
const webpack = require('webpack');
const TerserPlugin = require("terser-webpack-plugin");

function addCopyrightBannerPlugin(mode) {
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
  target = 'es5',
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
      addCopyrightBannerPlugin(mode),
    ],
    module: {
      rules: [{
        test: /\.tsx?$/,
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

module.exports = (env, argv) => {
  return [
    buildConfig({ suffix: '.es6', mode: 'development', devtool: 'inline-source-map', target: 'es6' }),
    buildConfig({ mode: 'development', devtool: 'inline-source-map' }),
    buildConfig({ suffix: '.es6.min', target: 'es6' }),
    buildConfig({ suffix: '.min' }),
    buildConfig({ format: 'commonjs', suffix: '.node' }),
  ];
};
