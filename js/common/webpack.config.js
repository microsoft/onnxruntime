// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

import webpack from 'webpack';
import {resolve} from 'node:path';
import {encode} from 'querystring';
import {DEFAULT_ES_VERSION, addCopyrightBannerPlugin} from '../webpack.shared.mjs';

const ifdef_training_opts = {
  TRAINING: true,
  "ifdef-fill-with-blanks": true
};

const ifdef_inference_opts = {
  TRAINING: false
};

// if training flag is used, then pass ifdef_training_opts

function buildConfig({
  suffix = '.js',                  // '.js', '.min.js', ...
  format = 'umd',                  // 'umd', 'commonjs'
  target = 'web',                  // 'web', 'node'
  esVersion = DEFAULT_ES_VERSION,  // 'es5', 'es6', ...
  mode = 'production',             // 'development', 'production'
  devtool = 'source-map',           // 'inline-source-map', 'source-map'
  ifdef_opts = ifdef_inference_opts,
}) {
  // output file name
  const filename = `ort-common${suffix}`;

  // variable name of the exported object.
  // - set to 'ort' when building 'umd' format.
  // - set to undefined when building other formats (commonjs/module)
  const exportName = format === 'umd' ? 'ort' : undefined;

  const ifdef_query = encode(ifdef_opts);

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
        },
        {
          loader: `ifdef-loader?${ifdef_query}`
        }]
      }]
    },
    mode,
    devtool,
  };
}

export default (env, argv) => {
  const ifdef_opts = env.t ? ifdef_training_opts : ifdef_inference_opts;
  return [
    buildConfig({suffix: '.es5.min.js', target: 'web', esVersion: 'es5', ifdef_opts: ifdef_opts}),
    buildConfig({suffix: '.min.js', ifdef_opts: ifdef_opts}),
    buildConfig({mode: 'development', devtool: 'inline-source-map', ifdef_opts: ifdef_opts}),
    buildConfig({
      suffix: '.node.cjs',
      target: 'node',
      format: 'commonjs',
    }),
  ];
};
