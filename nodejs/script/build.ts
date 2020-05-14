import {execSync, spawnSync} from 'child_process';
import * as fs from 'fs-extra';
import minimist from 'minimist';
import * as path from 'path';

// NPM configs (parsed via 'npm install --xxx')

// skip build on install. usually used in CI where build will be another step.
const SKIP = !!process.env.npm_config_ort_skip;
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
const REBUILD = !!buildArgs.rebuild;

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
if (CONFIG === 'Debug') {
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
