import {execSync, spawnSync} from 'child_process';
import * as fs from 'fs-extra';
import * as path from 'path';

// command line flags
const DEBUG = process.argv.slice(2).indexOf('--debug') !== -1;
const REBUILD = process.argv.slice(2).indexOf('--rebuild') !== -1;

// build path
const ROOT_FOLDER = path.join(__dirname, '..');
const BIN_FOLDER = path.join(ROOT_FOLDER, 'bin');

const NPM_BIN_FOLDER = execSync('npm bin', {encoding: 'utf8'}).trim();
const CMAKE_JS_FULL_PATH = path.join(NPM_BIN_FOLDER, 'cmake-js');

// if rebuild, clean up the dist folders
if (REBUILD) {
  fs.removeSync(BIN_FOLDER);
}

const command = CMAKE_JS_FULL_PATH;
const args = [(REBUILD ? 'rebuild' : 'compile'), '--arch=x64', '--CDnapi_build_version=3'];
if (DEBUG) {
  args.push('-D');
}

// launch cmake-js
const proc = spawnSync(command, args, {shell: true, stdio: 'inherit', cwd: ROOT_FOLDER});
if (proc.status !== 0) {
  if (proc.error) {
    console.error(proc.error);
  }
  process.exit(proc.status === null ? undefined : proc.status);
}
