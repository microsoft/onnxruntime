// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

import webpack from 'webpack';
import TerserPlugin from 'terser-webpack-plugin';
import {resolve, dirname} from 'node:path';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';

/**
 * ECMAScript version for default onnxruntime JavaScript API builds
 */
export const DEFAULT_ES_VERSION = 'es2017';

// how to use "__dirname" in ESM: https://shramko.dev/blog/dirname-error
const __dirname = dirname(fileURLToPath(import.meta.url));

const terserEcmaVersionFromWebpackEsVersion = (target) => {
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
};

const getPackageFullName = (name) => {
  switch (name) {
    case 'common':
      return `ONNX Runtime Common`;
    case 'node':
      return `ONNX Runtime Node.js Binding`;
    case 'web':
      return `ONNX Runtime Web`;
    case 'react-native':
      return `ONNX Runtime React-native`;
    default:
      throw new RangeError(`unknown package name: ${name}`);
  }
};

/**
 * Get package version by reading the file "package.json" under the package folder
 * @param {'common'|'node'|'web'|'react-native'} name - the package name
 * @returns a string representing the package version
 */
const getPackageVersion = (name) => {
  const normalizedName = name.replace('-', '_');
  const packageJsonFileContent = readFileSync(resolve(__dirname, normalizedName, 'package.json'));
  const packageJson = JSON.parse(packageJsonFileContent);
  return packageJson.version;
};

/**
 *
 * @param {'development'|'production'} mode - specify webpack build mode
 * @param {'common'|'node'|'web'|'react-native'} packageName - specify the name of the package
 * @param {'es5'|'es6'|'es2015'|'es2017'} esVersion - specify the ECMAScript version
 * @returns
 */
export const addCopyrightBannerPlugin = (mode, packageName, esVersion) => {
  const COPYRIGHT_BANNER = `/*!
 * ${getPackageFullName(packageName)} v${getPackageVersion(packageName)}
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */`;

  if (mode === 'production') {
    // in 'production' mode, webpack uses terser to minimize the code.
    // we set options.format.preamble to make sure terser generates correct copyright banner.
    return new TerserPlugin({
      extractComments: false,
      terserOptions: {
        ecma: terserEcmaVersionFromWebpackEsVersion(esVersion),
        format: {
          preamble: COPYRIGHT_BANNER,
          comments: false,
        },
        compress: {passes: 2}
      }
    });
  } else {
    // in 'development' mode, webpack does not minimize the code.
    // we use the webpack builtin plugin BannerPlugin to insert the banner.
    return new webpack.BannerPlugin({banner: COPYRIGHT_BANNER, raw: true});
  }
};
