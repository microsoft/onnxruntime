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
// --bundle-mode=node
const MODE = args['bundle-mode'] || 'prod';
if (['prod', 'dev', 'perf', 'node'].indexOf(MODE) === -1) {
  throw new Error(`unknown build mode: ${MODE}`);
}

// --wasm (default)
// --no-wasm
const WASM = typeof args.wasm === 'undefined' ? true : !!args.wasm;

// Path variables
const WASM_BINDING_FOLDER = path.join(__dirname, '..', 'lib', 'wasm', 'binding');
const WASM_BINDING_JS_PATH = path.join(WASM_BINDING_FOLDER, 'ort-wasm.js');
const WASM_BINDING_THREADED_JS_PATH = path.join(WASM_BINDING_FOLDER, 'ort-wasm-threaded.js');
const WASM_BINDING_THREADED_WORKER_JS_PATH = path.join(WASM_BINDING_FOLDER, 'ort-wasm-threaded.worker.js');
const WASM_DIST_FOLDER = path.join(__dirname, '..', 'dist');
const WASM_WASM_PATH = path.join(WASM_DIST_FOLDER, 'ort-wasm.wasm');
const WASM_THREADED_WASM_PATH = path.join(WASM_DIST_FOLDER, 'ort-wasm-threaded.wasm');
const WASM_THREADED_WORKER_JS_PATH = path.join(WASM_DIST_FOLDER, 'ort-wasm-threaded.worker.js');
const WASM_THREADED_JS_PATH = path.join(WASM_DIST_FOLDER, 'ort-wasm-threaded.js');

function validateFile(path: string): void {
  npmlog.info('Build', `Ensure file: ${path}`);
  if (!fs.pathExistsSync(path)) {
    throw new Error(`file does not exist: ${path}`);
  }
  if (fs.statSync(path).size === 0) {
    throw new Error(`file is empty: ${path}`);
  }
}

npmlog.info('Build.Bundle', 'Retrieving npm bin folder...');
const npmBin = execSync('npm bin', {encoding: 'utf8'}).trimRight();
npmlog.info('Build.Bundle', `Retrieving npm bin folder... DONE, folder: ${npmBin}`);

if (WASM) {
  npmlog.info('Build', 'Validating WebAssembly artifacts...');
  try {
    validateFile(WASM_BINDING_JS_PATH);
    validateFile(WASM_BINDING_THREADED_JS_PATH);
    validateFile(WASM_BINDING_THREADED_WORKER_JS_PATH);
    validateFile(WASM_WASM_PATH);
    validateFile(WASM_THREADED_WASM_PATH);
  } catch (e) {
    npmlog.error('Build', `WebAssembly files are not ready. build WASM first. ERR: ${e}`);
    throw e;
  }
  npmlog.info('Build', 'Validating WebAssembly artifacts... DONE');

  npmlog.info('Build', `Copying file "ort-wasm-threaded.js" to "${WASM_DIST_FOLDER}"...`);
  try {
    fs.copyFileSync(WASM_BINDING_THREADED_JS_PATH, WASM_THREADED_JS_PATH);
    validateFile(WASM_THREADED_JS_PATH);
  } catch (e) {
    npmlog.error('Build', `Failed to copy file. ERR: ${e}`);
    throw e;
  }
  npmlog.info('Build', `Copying file "ort-wasm-threaded.js" to "${WASM_DIST_FOLDER}"... DONE`);

  npmlog.info('Build', `Copying file "ort-wasm-threaded.worker.js" to "${WASM_DIST_FOLDER}"...`);
  try {
    fs.copyFileSync(WASM_BINDING_THREADED_WORKER_JS_PATH, WASM_THREADED_WORKER_JS_PATH);
    validateFile(WASM_THREADED_WORKER_JS_PATH);
  } catch (e) {
    npmlog.error('Build', `Failed to copy file. ERR: ${e}`);
    throw e;
  }
  npmlog.info('Build', `Copying file "ort-wasm-threaded.worker.js" to "${WASM_DIST_FOLDER}"... DONE`);

  npmlog.info('Build', 'Minimizing generated JavaScript files...');
  // const terserCommand = path.join(npmBin, 'terser');
  // npmlog.info('Build', `Running terser on file "${WASM_THREADED_JS_PATH}"`);
  // const terserArgsThreadedJs = ['--env', `--bundle-mode=${MODE}`];
  // npmlog.info('Build', `Running terser on file "${WASM_THREADED_WORKER_JS_PATH}"`);

  npmlog.info('Build', 'Minimizing generated JavaScript files... DONE');
}

npmlog.info('Build', 'Building bundle...');
{
  npmlog.info('Build.Bundle', 'Running webpack to generate bundles...');
  const webpackCommand = path.join(npmBin, 'webpack');
  const webpackArgs = ['--env', `--bundle-mode=${MODE}`];
  npmlog.info('Build.Bundle', `CMD: ${webpackCommand} ${webpackArgs.join(' ')}`);
  const webpack = spawnSync(webpackCommand, webpackArgs, {shell: true, stdio: 'inherit'});
  if (webpack.status !== 0) {
    console.error(webpack.error);
    process.exit(webpack.status === null ? undefined : webpack.status);
  }
  npmlog.info('Build.Bundle', 'Running webpack to generate bundles... DONE');
}
npmlog.info('Build', 'Building bundle... DONE');
