// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {execSync, spawnSync} from 'child_process';
import * as fs from 'fs-extra';
import minimist from 'minimist';
import npmlog from 'npmlog';
import * as path from 'path';

// CMD args
const args = minimist(process.argv);

// --bundle-mode=prod (default)
// --bundle-mode=dev
// --bundle-mode=perf
const MODE = args['bundle-mode'] || 'prod';
if (['prod', 'dev', 'perf'].indexOf(MODE) === -1) {
  throw new Error(`unknown build mode: ${MODE}`);
}

// --wasm (default)
// --no-wasm
const WASM = typeof args.wasm === 'undefined' ? true : !!args.wasm;

// Path variables
const WASM_BINDING_FOLDER = path.join(__dirname, '..', 'lib', 'wasm', 'binding');
const WASM_JS_PATH = path.join(WASM_BINDING_FOLDER, 'ort-wasm.js');
const WASM_THREADED_JS_PATH = path.join(WASM_BINDING_FOLDER, 'ort-wasm-threaded.js');
const WASM_DIST_FOLDER = path.join(__dirname, '..', 'dist');
const WASM_WASM_PATH = path.join(WASM_DIST_FOLDER, 'ort-wasm.wasm');
const WASM_THREADED_WASM_PATH = path.join(WASM_DIST_FOLDER, 'ort-wasm-threaded.wasm');
const WASM_THREADED_WORKER_JS_PATH = path.join(WASM_DIST_FOLDER, 'ort-wasm-threaded.worker.js');

function validateFile(path: string): void {
  npmlog.info('Build', `Ensure file: ${path}`);
  if (!fs.pathExistsSync(path)) {
    throw new Error(`file does not exist: ${path}`);
  }
  if (fs.statSync(path).size === 0) {
    throw new Error(`file is empty: ${path}`);
  }
}

if (WASM) {
  npmlog.info('Build', 'Validating WebAssembly artifacts...');
  try {
    validateFile(WASM_JS_PATH);
    validateFile(WASM_THREADED_JS_PATH);
    validateFile(WASM_WASM_PATH);
    validateFile(WASM_THREADED_WASM_PATH);
    validateFile(WASM_THREADED_WORKER_JS_PATH);
  } catch (e) {
    npmlog.error('Build', `WebAssembly files are not ready. build WASM first. ERR: ${e}`);
    throw e;
  }
  npmlog.info('Build', 'Validating WebAssembly artifacts... DONE');
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
