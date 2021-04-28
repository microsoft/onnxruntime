// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {execSync, spawnSync} from 'child_process';
import * as fs from 'fs-extra';
import minimist from 'minimist';
import * as path from 'path';

// NPM configs (parsed via 'npm install --xxx')

// skip build on install. usually used in CI where build will be another step.
const SKIP = !!process.env.npm_config_ort_skip_build;
if (SKIP) {
  process.exit(0);
}

// command line flags
const buildArgs = minimist(process.argv.slice(2));

// currently only support Debug, Release and RelWithDebInfo
const CONFIG: 'Debug'|'Release'|'RelWithDebInfo' = buildArgs.config || 'RelWithDebInfo';
if (CONFIG !== 'Debug' && CONFIG !== 'Release' && CONFIG !== 'RelWithDebInfo') {
  throw new Error(`unrecognized config: ${CONFIG}`);
}
const ONNXRUNTIME_BUILD_DIR = buildArgs['onnxruntime-build-dir'];
const REBUILD = !!buildArgs.rebuild;

// build path
const ROOT_FOLDER = path.join(__dirname, '..');
const BIN_FOLDER = path.join(ROOT_FOLDER, 'bin');
const BUILD_FOLDER = path.join(ROOT_FOLDER, 'build');

const NPM_BIN_FOLDER = execSync('npm bin', {encoding: 'utf8'}).trim();
const CMAKE_JS_FULL_PATH = path.join(NPM_BIN_FOLDER, 'cmake-js');

// if rebuild, clean up the dist folders
if (REBUILD) {
  fs.removeSync(BIN_FOLDER);
  fs.removeSync(BUILD_FOLDER);
}

const command = CMAKE_JS_FULL_PATH;
const args = [
  (REBUILD ? 'reconfigure' : 'configure'),
  '--arch=x64',
  '--CDnapi_build_version=3',
  `--CDCMAKE_BUILD_TYPE=${CONFIG}`,
];
if (ONNXRUNTIME_BUILD_DIR && typeof ONNXRUNTIME_BUILD_DIR === 'string') {
  args.push(`--CDONNXRUNTIME_BUILD_DIR=${ONNXRUNTIME_BUILD_DIR}`);
}

// launch cmake-js configure
const procCmakejs = spawnSync(command, args, {shell: true, stdio: 'inherit', cwd: ROOT_FOLDER});
if (procCmakejs.status !== 0) {
  if (procCmakejs.error) {
    console.error(procCmakejs.error);
  }
  process.exit(procCmakejs.status === null ? undefined : procCmakejs.status);
}

// launch cmake to build
const procCmake =
    spawnSync('cmake', ['--build', '.', '--config', CONFIG], {shell: true, stdio: 'inherit', cwd: BUILD_FOLDER});
if (procCmake.status !== 0) {
  if (procCmake.error) {
    console.error(procCmake.error);
  }
  process.exit(procCmake.status === null ? undefined : procCmake.status);
}
