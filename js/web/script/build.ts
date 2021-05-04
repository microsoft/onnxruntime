// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {execSync, spawnSync} from 'child_process';
import * as fs from 'fs-extra';
import minimist from 'minimist';
import npmlog from 'npmlog';
import * as path from 'path';

// CMD args
const args = minimist(process.argv);
const MODE = args.config || 'prod';  // prod|dev|test
if (['prod', 'dev', 'test'].indexOf(MODE) === -1) {
  throw new Error(`unknown build mode: ${MODE}`);
}

// Path variables
const WASM_BINDING_FOLDER = path.join(__dirname, '..', 'lib', 'wasm', 'binding');
const WASM_JS_PATH = path.join(WASM_BINDING_FOLDER, 'ort-wasm.js');
const WASM_DIST_FOLDER = path.join(__dirname, '..', 'dist');
const WASM_DIST_PATH = path.join(WASM_DIST_FOLDER, 'ort-wasm.wasm');

try {
  npmlog.info('Build', `Ensure file: ${WASM_JS_PATH}`);
  if (!fs.pathExistsSync(WASM_JS_PATH)) {
    throw new Error(`file does not exist: ${WASM_JS_PATH}`);
  }
  npmlog.info('Build', `Ensure file: ${WASM_DIST_PATH}`);
  if (!fs.pathExistsSync(WASM_DIST_PATH)) {
    throw new Error(`file does not exist: ${WASM_DIST_PATH}`);
  }
} catch (e) {
  npmlog.error('Build', `WebAssembly files are not ready. build WASM first. ERR: ${e}`);
  throw e;
}

npmlog.info('Build', 'Building bundle...');
{
  npmlog.info('Build.Bundle', '(1/2) Retrieving npm bin folder...');
  const npmBin = execSync('npm bin', {encoding: 'utf8'}).trimRight();
  npmlog.info('Build.Bundle', `(1/2) Retrieving npm bin folder... DONE, folder: ${npmBin}`);

  npmlog.info('Build.Bundle', '(2/2) Running webpack to generate bundles...');
  const webpackCommand = path.join(npmBin, 'webpack');
  const webpackArgs = ['--env', `--bundle-mode=${MODE}`];
  npmlog.info('Build.Bundle', `CMD: ${webpackCommand} ${webpackArgs.join(' ')}`);
  const webpack = spawnSync(webpackCommand, webpackArgs, {shell: true, stdio: 'inherit'});
  if (webpack.status !== 0) {
    console.error(webpack.error);
    process.exit(webpack.status === null ? undefined : webpack.status);
  }
  npmlog.info('Build.Bundle', '(2/2) Running webpack to generate bundles... DONE');
}
npmlog.info('Build', 'Building bundle... DONE');
