// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {spawnSync} from 'child_process';
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

// -a; --analyzer
const ANALYZER = !!args.a || !!args.analyzer;

// -f; --filter=<regex>
const FILTER = args.f || args.filter;

// Path variables
const ROOT_FOLDER = path.join(__dirname, '..');
const WASM_BINDING_FOLDER = path.join(ROOT_FOLDER, 'lib', 'wasm', 'binding');
const WASM_BINDING_JS_PATH = path.join(WASM_BINDING_FOLDER, 'ort-wasm.js');
const WASM_BINDING_THREADED_JS_PATH = path.join(WASM_BINDING_FOLDER, 'ort-wasm-threaded.js');
const WASM_BINDING_SIMD_THREADED_JSEP_JS_PATH = path.join(WASM_BINDING_FOLDER, 'ort-wasm-simd-threaded.jsep.js');
const WASM_BINDING_THREADED_WORKER_JS_PATH = path.join(WASM_BINDING_FOLDER, 'ort-wasm-threaded.worker.js');
const WASM_BINDING_THREADED_MIN_JS_PATH = path.join(WASM_BINDING_FOLDER, 'ort-wasm-threaded.min.js');
const WASM_BINDING_SIMD_THREADED_JSEP_MIN_JS_PATH =
    path.join(WASM_BINDING_FOLDER, 'ort-wasm-simd-threaded.jsep.min.js');
const WASM_BINDING_THREADED_MIN_WORKER_JS_PATH = path.join(WASM_BINDING_FOLDER, 'ort-wasm-threaded.min.worker.js');

const WASM_DIST_FOLDER = path.join(ROOT_FOLDER, 'dist');
const WASM_WASM_PATH = path.join(WASM_DIST_FOLDER, 'ort-wasm.wasm');
const WASM_THREADED_WASM_PATH = path.join(WASM_DIST_FOLDER, 'ort-wasm-threaded.wasm');
const WASM_SIMD_WASM_PATH = path.join(WASM_DIST_FOLDER, 'ort-wasm-simd.wasm');
const WASM_SIMD_THREADED_WASM_PATH = path.join(WASM_DIST_FOLDER, 'ort-wasm-simd-threaded.wasm');
const WASM_SIMD_JSEP_WASM_PATH = path.join(WASM_DIST_FOLDER, 'ort-wasm-simd.jsep.wasm');
const WASM_SIMD_THREADED_JSEP_WASM_PATH = path.join(WASM_DIST_FOLDER, 'ort-wasm-simd-threaded.jsep.wasm');
const WASM_THREADED_WORKER_JS_PATH = path.join(WASM_DIST_FOLDER, 'ort-wasm-threaded.worker.js');
const WASM_THREADED_JS_PATH = path.join(WASM_DIST_FOLDER, 'ort-wasm-threaded.js');
const WASM_SIMD_THREADED_JSEP_JS_PATH = path.join(WASM_DIST_FOLDER, 'ort-wasm-simd-threaded.jsep.js');

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
    validateFile(WASM_BINDING_JS_PATH);
    validateFile(WASM_BINDING_THREADED_JS_PATH);
    validateFile(WASM_BINDING_SIMD_THREADED_JSEP_JS_PATH);
    validateFile(WASM_BINDING_THREADED_WORKER_JS_PATH);
    validateFile(WASM_WASM_PATH);
    validateFile(WASM_THREADED_WASM_PATH);
    validateFile(WASM_SIMD_WASM_PATH);
    validateFile(WASM_SIMD_THREADED_WASM_PATH);
    validateFile(WASM_SIMD_JSEP_WASM_PATH);
    validateFile(WASM_SIMD_THREADED_JSEP_WASM_PATH);
  } catch (e) {
    npmlog.error('Build', `WebAssembly files are not ready. build WASM first. ERR: ${e}`);
    throw e;
  }
  npmlog.info('Build', 'Validating WebAssembly artifacts... DONE');

  const VERSION = require(path.join(__dirname, '../package.json')).version;
  const COPYRIGHT_BANNER = `/*!
 * ONNX Runtime Web v${VERSION}
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
`;

  npmlog.info('Build', 'Minimizing file "ort-wasm-threaded.js"...');
  try {
    const terser = spawnSync(
        'npx',
        [
          'terser', WASM_BINDING_THREADED_JS_PATH, '--compress', 'passes=2', '--format', 'comments=false', '--mangle',
          'reserved=[_scriptDir,startWorker]', '--module'
        ],
        {shell: true, encoding: 'utf-8', cwd: ROOT_FOLDER});
    if (terser.status !== 0) {
      console.error(terser.error);
      process.exit(terser.status === null ? undefined : terser.status);
    }

    fs.writeFileSync(WASM_BINDING_THREADED_MIN_JS_PATH, terser.stdout);
    fs.writeFileSync(WASM_THREADED_JS_PATH, `${COPYRIGHT_BANNER}${terser.stdout}`);

    validateFile(WASM_BINDING_THREADED_MIN_JS_PATH);
    validateFile(WASM_THREADED_JS_PATH);
  } catch (e) {
    npmlog.error('Build', `Failed to run terser on ort-wasm-threaded.js. ERR: ${e}`);
    throw e;
  }
  npmlog.info('Build', 'Minimizing file "ort-wasm-threaded.js"... DONE');

  npmlog.info('Build', 'Minimizing file "ort-wasm-simd-threaded.jsep.js"...');
  try {
    const terser = spawnSync(
        'npx',
        [
          'terser', WASM_BINDING_SIMD_THREADED_JSEP_JS_PATH, '--compress', 'passes=2', '--format', 'comments=false',
          '--mangle', 'reserved=[_scriptDir,startWorker]', '--module'
        ],
        {shell: true, encoding: 'utf-8', cwd: ROOT_FOLDER});
    if (terser.status !== 0) {
      console.error(terser.error);
      process.exit(terser.status === null ? undefined : terser.status);
    }

    fs.writeFileSync(WASM_BINDING_SIMD_THREADED_JSEP_MIN_JS_PATH, terser.stdout);
    fs.writeFileSync(WASM_SIMD_THREADED_JSEP_JS_PATH, `${COPYRIGHT_BANNER}${terser.stdout}`);

    validateFile(WASM_BINDING_SIMD_THREADED_JSEP_MIN_JS_PATH);
    validateFile(WASM_SIMD_THREADED_JSEP_JS_PATH);
  } catch (e) {
    npmlog.error('Build', `Failed to run terser on ort-wasm-threaded.js. ERR: ${e}`);
    throw e;
  }
  npmlog.info('Build', 'Minimizing file "ort-wasm-simd-threaded.jsep.js"... DONE');

  npmlog.info('Build', 'Minimizing file "ort-wasm-threaded.worker.js"...');
  try {
    const terser = spawnSync(
        'npx',
        [
          'terser', WASM_BINDING_THREADED_WORKER_JS_PATH, '--compress', 'passes=2', '--format', 'comments=false',
          '--mangle', 'reserved=[_scriptDir,startWorker]', '--toplevel'
        ],
        {shell: true, encoding: 'utf-8'});
    if (terser.status !== 0) {
      console.error(terser.error);
      process.exit(terser.status === null ? undefined : terser.status);
    }

    fs.writeFileSync(WASM_BINDING_THREADED_MIN_WORKER_JS_PATH, terser.stdout);
    fs.writeFileSync(WASM_THREADED_WORKER_JS_PATH, `${COPYRIGHT_BANNER}${terser.stdout}`);

    validateFile(WASM_BINDING_THREADED_MIN_WORKER_JS_PATH);
    validateFile(WASM_THREADED_WORKER_JS_PATH);
  } catch (e) {
    npmlog.error('Build', `Failed to run terser on ort-wasm-threaded.worker.js. ERR: ${e}`);
    throw e;
  }
  npmlog.info('Build', 'Minimizing file "ort-wasm-threaded.worker.js"... DONE');
}

npmlog.info('Build', 'Building bundle...');
{
  npmlog.info('Build.Bundle', 'Running webpack to generate bundles...');
  const webpackArgs = ['webpack', '--env', `--bundle-mode=${MODE}`];
  if (ANALYZER) {
    webpackArgs.push('--env', '-a');
  }
  if (FILTER) {
    webpackArgs.push('--env', `-f=${FILTER}`);
  }
  npmlog.info('Build.Bundle', `CMD: npx ${webpackArgs.join(' ')}`);
  const webpack = spawnSync('npx', webpackArgs, {shell: true, stdio: 'inherit', cwd: ROOT_FOLDER});
  if (webpack.status !== 0) {
    console.error(webpack.error);
    process.exit(webpack.status === null ? undefined : webpack.status);
  }
  npmlog.info('Build.Bundle', 'Running webpack to generate bundles... DONE');
}
npmlog.info('Build', 'Building bundle... DONE');
