// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {execSync, spawnSync} from 'child_process';
import * as fs from 'fs-extra';
import minimist from 'minimist';
import * as os from 'os';
import * as path from 'path';

// command line flags
const buildArgs = minimist(process.argv.slice(2));

// --config=Debug|Release|RelWithDebInfo
const CONFIG: 'Debug'|'Release'|'RelWithDebInfo' =
    buildArgs.config || (os.platform() === 'win32' ? 'RelWithDebInfo' : 'Release');
if (CONFIG !== 'Debug' && CONFIG !== 'Release' && CONFIG !== 'RelWithDebInfo') {
  throw new Error(`unrecognized config: ${CONFIG}`);
}
// --arch=x64|ia32|arm64|arm
const ARCH: 'x64'|'ia32'|'arm64'|'arm' = buildArgs.arch || os.arch();
if (ARCH !== 'x64' && ARCH !== 'ia32' && ARCH !== 'arm64' && ARCH !== 'arm') {
  throw new Error(`unrecognized architecture: ${ARCH}`);
}
// --onnxruntime-build-dir=
const ONNXRUNTIME_BUILD_DIR = buildArgs['onnxruntime-build-dir'];
// --rebuild
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
  `--arch=${ARCH}`,
  '--CDnapi_build_version=3',
  `--CDCMAKE_BUILD_TYPE=${CONFIG}`,
];
if (ONNXRUNTIME_BUILD_DIR && typeof ONNXRUNTIME_BUILD_DIR === 'string') {
  args.push(`--CDONNXRUNTIME_BUILD_DIR=${ONNXRUNTIME_BUILD_DIR}`);
}

// set CMAKE_OSX_ARCHITECTURES for macOS build
if (os.platform() === 'darwin') {
  if (ARCH === 'x64') {
    args.push('--CDCMAKE_OSX_ARCHITECTURES=x86_64');
  } else if (ARCH === 'arm64') {
    args.push('--CDCMAKE_OSX_ARCHITECTURES=arm64');
  } else {
    throw new Error(`architecture not supported for macOS build: ${ARCH}`);
  }
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
