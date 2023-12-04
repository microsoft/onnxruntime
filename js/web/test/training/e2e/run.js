// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const path = require('path');
const fs = require('fs-extra');
const {spawn} = require('child_process');
const startServer = require('./simple-http-server');
const minimist = require('minimist');

// copy whole folder to out-side of <ORT_ROOT>/js/ because we need to test in a folder that no `package.json` file
// exists in its parent folder.
// here we use <ORT_ROOT>/build/js/e2e/ for the test

const TEST_E2E_SRC_FOLDER = __dirname;
const JS_ROOT_FOLDER = path.resolve(__dirname, '../../../..');
const TEST_E2E_RUN_FOLDER = path.resolve(JS_ROOT_FOLDER, '../build/js/e2e');
const NPM_CACHE_FOLDER = path.resolve(TEST_E2E_RUN_FOLDER, '../npm_cache');
const CHROME_USER_DATA_FOLDER = path.resolve(TEST_E2E_RUN_FOLDER, '../user_data');
fs.emptyDirSync(TEST_E2E_RUN_FOLDER);
fs.emptyDirSync(NPM_CACHE_FOLDER);
fs.emptyDirSync(CHROME_USER_DATA_FOLDER);
fs.copySync(TEST_E2E_SRC_FOLDER, TEST_E2E_RUN_FOLDER);

// training data to copy
const ORT_ROOT_FOLDER = path.resolve(JS_ROOT_FOLDER, '..');
const TRAINING_DATA_FOLDER = path.resolve(ORT_ROOT_FOLDER, 'onnxruntime/test/testdata/training_api');
const TRAININGDATA_DEST = path.resolve(TEST_E2E_SRC_FOLDER, 'data');

// always use a new folder as user-data-dir
let nextUserDataDirId = 0;
function getNextUserDataDir() {
  const dir = path.resolve(CHROME_USER_DATA_FOLDER, nextUserDataDirId.toString())
  nextUserDataDirId++;
  fs.emptyDirSync(dir);
  return dir;
}

// commandline arguments
const BROWSER = minimist(process.argv.slice(2)).browser || 'Chrome_default';

async function main() {
  // find packed package
  const {globbySync} = await import('globby');

  const ORT_COMMON_FOLDER = path.resolve(JS_ROOT_FOLDER, 'common');
  const ORT_COMMON_PACKED_FILEPATH_CANDIDATES = globbySync('onnxruntime-common-*.tgz', {cwd: ORT_COMMON_FOLDER});

  const PACKAGES_TO_INSTALL = [];

  if (ORT_COMMON_PACKED_FILEPATH_CANDIDATES.length === 1) {
    PACKAGES_TO_INSTALL.push(path.resolve(ORT_COMMON_FOLDER, ORT_COMMON_PACKED_FILEPATH_CANDIDATES[0]));
  } else if (ORT_COMMON_PACKED_FILEPATH_CANDIDATES.length > 1) {
    throw new Error('multiple packages found for onnxruntime-common.');
  }

  const ORT_WEB_FOLDER = path.resolve(JS_ROOT_FOLDER, 'web');
  const ORT_WEB_PACKED_FILEPATH_CANDIDATES = globbySync('onnxruntime-web-*.tgz', {cwd: ORT_WEB_FOLDER});
  if (ORT_WEB_PACKED_FILEPATH_CANDIDATES.length !== 1) {
    throw new Error('cannot find exactly single package for onnxruntime-web.');
  }
  PACKAGES_TO_INSTALL.push(path.resolve(ORT_WEB_FOLDER, ORT_WEB_PACKED_FILEPATH_CANDIDATES[0]));

  // we start here:

  // install dev dependencies
  await runInShell(`npm install`);

  // npm install with "--cache" to install packed packages with an empty cache folder
  await runInShell(`npm install --cache "${NPM_CACHE_FOLDER}" ${PACKAGES_TO_INSTALL.map(i => `"${i}"`).join(' ')}`);

  // prepare .wasm files for path override testing
  prepareWasmPathOverrideFiles();

  // prepare training data
  await prepareTrainingDataByCopying();

  // test cases with self-host (ort hosted in same origin)
  await testAllBrowserCases({hostInKarma: true});

  // test cases without self-host (ort hosted in same origin)
  startServer(path.resolve(TEST_E2E_RUN_FOLDER, 'node_modules', 'onnxruntime-web'));
  await testAllBrowserCases({hostInKarma: false});

  // no error occurs, exit with code 0
  process.exit(0);
}

function prepareWasmPathOverrideFiles() {
  const folder = path.join(TEST_E2E_RUN_FOLDER, 'test-wasm-path-override');
  const sourceFile = path.join(TEST_E2E_RUN_FOLDER, 'node_modules', 'onnxruntime-web', 'dist', 'ort-training-wasm-simd.wasm');
  fs.emptyDirSync(folder);
  fs.copyFileSync(sourceFile, path.join(folder, 'ort-training-wasm.wasm'));
  fs.copyFileSync(sourceFile, path.join(folder, 'renamed.wasm'));
}

async function testAllBrowserCases({hostInKarma}) {
  await runKarma({hostInKarma, main: './browser-test-wasm.js'});
}

async function runKarma({hostInKarma, main, browser = BROWSER, ortMain = 'ort.training.wasm.min.js'}) {
  console.log('Running karma =======================');
  console.log(ortMain);
  console.log('=======================');
  const selfHostFlag = hostInKarma ? '--self-host' : '';
  await runInShell(`npx karma start --single-run --browsers ${browser} ${selfHostFlag} --ort-main=${
      ortMain} --test-main=${main} --user-data=${getNextUserDataDir()}`);
}

async function runInShell(cmd) {
  console.log('===============================================================');
  console.log(' Running command in shell:');
  console.log(' > ' + cmd);
  console.log('===============================================================');
  let complete = false;
  const childProcess = spawn(cmd, {shell: true, stdio: 'inherit', cwd: TEST_E2E_RUN_FOLDER});
  childProcess.on('close', function(code) {
    if (code !== 0) {
      process.exit(code);
    } else {
      complete = true;
    }
  });
  while (!complete) {
    await delay(100);
  }
}

async function delay(ms) {
  return new Promise(function(resolve) {
    setTimeout(function() {
      resolve();
    }, ms);
  });
}

function prepareTrainingDataByCopying() {
  const dirs = fs.readdirSync(TRAINING_DATA_FOLDER);

  for (let i = 0; i < dirs.length; i++) {
    const d = dirs[i];
    if (fs.statSync(path.join(TRAINING_DATA_FOLDER, d)).isFile()) {
      const src = path.join(TRAINING_DATA_FOLDER, d);
      const dest = path.join(TRAININGDATA_DEST, d);
      fs.copyFile(src, dest);
      console.log(`Copied ${src} to ${dest}`);
    }
  }
}

main();
