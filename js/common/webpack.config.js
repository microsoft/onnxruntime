// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

import webpack from 'webpack';
import {resolve} from 'node:path';
import {DEFAULT_ES_VERSION, addCopyrightBannerPlugin} from '../webpack.shared.mjs';

function buildConfig({
  suffix = '.js',                  // '.js', '.min.js', ...
  format = 'umd',                  // 'umd', 'commonjs'
  target = 'web',                  // 'web', 'node'
  esVersion = DEFAULT_ES_VERSION,  // 'es5', 'es6', ...
  mode = 'production',             // 'development', 'production'
  devtool = 'source-map'           // 'inline-source-map', 'source-map'
}) {
  // output file name
  const filename = `ort-common${suffix}`;

  // variable name of the exported object.
  // - set to 'ort' when building 'umd' format.
  // - set to undefined when building other formats (commonjs/module)
  const exportName = format === 'umd' ? 'ort' : undefined;

  return {
    target: [target, esVersion],
    entry: resolve('./lib/index.ts'),
    output: {
      path: resolve('./dist'),
      filename,
      library: {name: exportName, type: format},
    },
    resolve: {
      extensions: ['.ts', '.js'],
      extensionAlias: {'.js': ['.ts', '.js']},
    },
    plugins: [
      new webpack.WatchIgnorePlugin({paths: [/\.js$/, /\.d\.ts$/]}),
      addCopyrightBannerPlugin(mode, 'common', esVersion),
    ],
    module: {
      rules: [{
        test: /\.ts$/,
        use: [{
          loader: 'ts-loader',
          options: {compilerOptions: {target: esVersion}},
        }]
      }]
    },
    mode,
    devtool,
  };
}

export default (env, argv) => {
  return [
    buildConfig({suffix: '.es5.min.js', target: 'web', esVersion: 'es5'}),
    buildConfig({suffix: '.min.js'}),
    buildConfig({mode: 'development', devtool: 'inline-source-map'}),
    buildConfig({
      suffix: '.node.cjs',
      target: 'node',
      format: 'commonjs',
    }),
  ];
};
