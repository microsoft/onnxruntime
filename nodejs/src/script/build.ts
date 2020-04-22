import {execSync, spawnSync} from 'child_process';
import * as fs from 'fs-extra';
import * as os from 'os';
import * as path from 'path';

// command line flags
const DEBUG = process.argv.slice(2).indexOf('--debug') !== -1;
const REBUILD = process.argv.slice(2).indexOf('--rebuild') !== -1;

// configs
const BUILD_TYPE = DEBUG ? 'Debug' : 'Release';

// build paths
const ROOT = path.join(__dirname, '..');
const BIN = path.join(ROOT, 'bin', 'napi-v4');
const NPM_BIN_FOLDER = execSync('npm bin', {encoding: 'utf8'}).trim();
const CMAKE_JS_FULL_PATH = path.join(NPM_BIN_FOLDER, 'cmake-js');
const BUILD_OUTPUT_FOLDER = path.join(ROOT, 'build', BUILD_TYPE);

// onnxruntime header files
const ONNXRUNTIME_HEADERS_FOLDER = path.join(ROOT, '../..', 'include/onnxruntime/core/session');
const ONNXRUNTIME_HEADERS_FILES = [
  path.join(ONNXRUNTIME_HEADERS_FOLDER, 'onnxruntime_cxx_api.h'),
];
// onnxruntime build artifacts
const ONNXRUNTIME_DIST_FOLDER =
    path.join(ROOT, '../../build', os.platform() === 'win32' ? `Windows/${BUILD_TYPE}/${BUILD_TYPE}` : BUILD_TYPE);
const ONNXRUNTIME_DIST_FILES = os.platform() === 'win32' ?
    [
      // windows dist files
      path.join(ONNXRUNTIME_DIST_FOLDER, 'onnxruntime.dll'),
    ] :
    os.platform() === 'darwin' ?
    [
      // macos dist files
      path.join(ONNXRUNTIME_DIST_FOLDER, 'onnxruntime.dylib'),
    ] :
    [
      // linux dist files
      path.join(ONNXRUNTIME_DIST_FOLDER, 'onnxruntime.so'),
    ];

if (os.platform() === 'win32' && DEBUG) {
  ONNXRUNTIME_DIST_FILES.push(path.join(ONNXRUNTIME_DIST_FOLDER, 'onnxruntime.pdb'));
}

// ====================

console.log('BUILD [1/3] verify onnxruntime files ...');

// check header files
for (const file of ONNXRUNTIME_HEADERS_FILES) {
  if (!fs.existsSync(file)) {
    throw new Error(`header file does not exist: ${file}`);
  }
}

// check dist files
for (const file of ONNXRUNTIME_DIST_FILES) {
  if (!fs.existsSync(file)) {
    throw new Error(`dist file does not exist: ${file}`);
  }
}

console.log('BUILD [2/3] build node binding ...');

const cmakejsArgs = [(REBUILD ? 'rebuild' : 'compile')];
if (DEBUG) {
  cmakejsArgs.push('-D');
}

const cmakejs = spawnSync(CMAKE_JS_FULL_PATH, cmakejsArgs, {shell: true, stdio: 'inherit'});
if (cmakejs.status !== 0) {
  if (cmakejs.error) {
    console.error(cmakejs.error);
  }
  process.exit(cmakejs.status === null ? undefined : cmakejs.status);
}

console.log('BUILD [3/3] binplace build artifacts ...');

fs.emptyDirSync(BIN);
fs.emptyDirSync(path.join(BIN, 'cpu'));

fs.copySync(
    path.join(BUILD_OUTPUT_FOLDER, 'onnxruntime_binding.node'), path.join(BIN, 'cpu', 'onnxruntime_binding.node'));
if (os.platform() === 'win32' && DEBUG) {
  fs.copySync(
      path.join(BUILD_OUTPUT_FOLDER, 'onnxruntime_binding.pdb'), path.join(BIN, 'cpu', 'onnxruntime_binding.pdb'));
}

for (const file of ONNXRUNTIME_DIST_FILES) {
  fs.copySync(file, path.join(BIN, 'cpu', path.basename(file)));
}
