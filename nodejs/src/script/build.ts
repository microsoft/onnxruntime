import {execSync, spawnSync} from 'child_process';
import * as fs from 'fs-extra';
import * as path from 'path';

// command line flags
const DEBUG = process.argv.slice(2).indexOf('--debug') !== -1;
const REBUILD = process.argv.slice(2).indexOf('--rebuild') !== -1;
const USE_PREBUILD = process.argv.slice(2).indexOf('--prebuild') !== -1;

// build path
const ROOT_FOLDER = path.join(__dirname, '..');
const BIN_FOLDER = path.join(ROOT_FOLDER, 'bin');
const PREBUILDS_FOLDER = path.join(ROOT_FOLDER, 'prebuilds');

const NPM_BIN_FOLDER = execSync('npm bin', {encoding: 'utf8'}).trim();
const CMAKE_JS_FULL_PATH = path.join(NPM_BIN_FOLDER, 'cmake-js');
const PREBUILD_FULL_PATH = path.join(NPM_BIN_FOLDER, 'prebuild');

// if rebuild, clean up the dist folders
if (REBUILD) {
  fs.removeSync(BIN_FOLDER);
  if (USE_PREBUILD) {
    fs.removeSync(PREBUILDS_FOLDER);
  }
}

let command: string;
let args: string[];
if (USE_PREBUILD) {
  command = PREBUILD_FULL_PATH;
  args = ['--backend', 'cmake-js', '--runtime', 'napi', '--include-regex', '"\\.+"', '--prepack', '"npm test"'];
  if (DEBUG) {
    args.push('--debug');
  }

} else {
  command = CMAKE_JS_FULL_PATH;
  args = [(REBUILD ? 'rebuild' : 'compile'), '--arch=x64', '--CDnapi_build_version=3'];
  if (DEBUG) {
    args.push('-D');
  }
}

// launch cmake-js/prebuild
const proc = spawnSync(command, args, {shell: true, stdio: 'inherit', cwd: ROOT_FOLDER});
if (proc.status !== 0) {
  if (proc.error) {
    console.error(proc.error);
  }
  process.exit(proc.status === null ? undefined : proc.status);
}
