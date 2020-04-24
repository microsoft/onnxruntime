import {execSync, spawnSync} from 'child_process';
import * as path from 'path';

// command line flags
const DEBUG = process.argv.slice(2).indexOf('--debug') !== -1;
const REBUILD = process.argv.slice(2).indexOf('--rebuild') !== -1;
const USE_PREBUILD = process.argv.slice(2).indexOf('--prebuild') !== -1;

// cmake-js path
const NPM_BIN_FOLDER = execSync('npm bin', {encoding: 'utf8'}).trim();
const CMAKE_JS_FULL_PATH = path.join(NPM_BIN_FOLDER, 'cmake-js');
const PREBUILD_FULL_PATH = path.join(NPM_BIN_FOLDER, 'prebuild');

let command: string;
let args: string[];
if (USE_PREBUILD) {
  command = PREBUILD_FULL_PATH;
  args = ['--backend', 'cmake-js', '--runtime', 'napi', '--include-regex', '"\\.(node|dll|pdb)$"'];
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

// launch cmake-js
const proc = spawnSync(command, args, {shell: true, stdio: 'inherit'});
if (proc.status !== 0) {
  if (proc.error) {
    console.error(proc.error);
  }
  process.exit(proc.status === null ? undefined : proc.status);
}
