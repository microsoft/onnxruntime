const path = require('path');
const fs = require('fs-extra');
const globby = require('globby');
const { spawn } = require('child_process');
const startServer = require('./simple-http-server');

// copy whole folder to out-side of <ORT_ROOT>/js/ because we need to test in a folder that no `package.json` file
// exists in its parent folder.
// here we use <ORT_ROOT>/build/js/e2e/ for the test

const TEST_E2E_SRC_FOLDER = __dirname;
const JS_ROOT_FOLDER = path.resolve(__dirname, '../../..');
const TEST_E2E_RUN_FOLDER = path.resolve(JS_ROOT_FOLDER, '../build/js/e2e');
const NPM_CACHE_FOLDER = path.resolve(TEST_E2E_RUN_FOLDER, '../npm_cache');
const CHROME_USER_DATA_FOLDER = path.resolve(TEST_E2E_RUN_FOLDER, '../user_data');
fs.emptyDirSync(TEST_E2E_RUN_FOLDER);
fs.emptyDirSync(NPM_CACHE_FOLDER);
fs.emptyDirSync(CHROME_USER_DATA_FOLDER);
fs.copySync(TEST_E2E_SRC_FOLDER, TEST_E2E_RUN_FOLDER);

// find packed package

const ORT_COMMON_FOLDER = path.resolve(JS_ROOT_FOLDER, 'common');
const ORT_COMMON_PACKED_FILEPATH_CANDIDATES = globby.sync('onnxruntime-common-*.tgz', { cwd: ORT_COMMON_FOLDER });
if (ORT_COMMON_PACKED_FILEPATH_CANDIDATES.length !== 1) {
  throw new Error('cannot find exactly single package for onnxruntime-common.');
}
const ORT_COMMON_PACKED_FILEPATH = path.resolve(ORT_COMMON_FOLDER, ORT_COMMON_PACKED_FILEPATH_CANDIDATES[0]);

const ORT_WEB_FOLDER = path.resolve(JS_ROOT_FOLDER, 'web');
const ORT_WEB_PACKED_FILEPATH_CANDIDATES = globby.sync('onnxruntime-web-*.tgz', { cwd: ORT_WEB_FOLDER });
if (ORT_WEB_PACKED_FILEPATH_CANDIDATES.length !== 1) {
  throw new Error('cannot find exactly single package for onnxruntime-web.');
}
const ORT_WEB_PACKED_FILEPATH = path.resolve(ORT_WEB_FOLDER, ORT_WEB_PACKED_FILEPATH_CANDIDATES[0]);

// we start here:

async function main() {
  // install dev dependencies
  await runInShell(`npm install"`);

  // npm install with "--cache" to install packed packages with an empty cache folder
  await runInShell(`npm install --cache "${NPM_CACHE_FOLDER}" "${ORT_COMMON_PACKED_FILEPATH}" "${ORT_WEB_PACKED_FILEPATH}"`);

  // test case run in Node.js
  await testAllNodejsCases();

  // test cases with self-host (ort hosted in same origin)
  await testAllBrowserCases({ hostInKarma: true });

  // test cases without self-host (ort hosted in same origin)
  startServer(path.resolve(TEST_E2E_RUN_FOLDER, 'node_modules', 'onnxruntime-web'));
  await testAllBrowserCases({ hostInKarma: false });

  // no error occurs, exit with code 0
  process.exit(0);
}

async function testAllNodejsCases() {
  await runInShell('node ./node_modules/mocha/bin/mocha ./node-test-main-no-threads.js');
  await runInShell('node ./node_modules/mocha/bin/mocha ./node-test-main.js');
  await runInShell('node --experimental-wasm-threads --experimental-wasm-bulk-memory ./node_modules/mocha/bin/mocha ./node-test-main-no-threads.js');
  await runInShell('node --experimental-wasm-threads --experimental-wasm-bulk-memory ./node_modules/mocha/bin/mocha ./node-test-main.js');
}

async function testAllBrowserCases({ hostInKarma }) {
  await runKarma({ hostInKarma, main: './browser-test-webgl.js', browser: 'Chrome_default' });
  await runKarma({ hostInKarma, main: './browser-test-wasm.js', browser: 'Chrome_default' });
  await runKarma({ hostInKarma, main: './browser-test-wasm-no-threads.js', browser: 'Chrome_default' });
}

async function runKarma({ hostInKarma, main, browser }) {
  const selfHostFlag = hostInKarma ? '--self-host' : '';
  await runInShell(
    `npx karma start --single-run --browsers ${browser} ${selfHostFlag} --test-main=${main} --user-data=${CHROME_USER_DATA_FOLDER}`);
}

async function runInShell(cmd) {
  console.log('===============================================================');
  console.log(' Running command in shell:');
  console.log(' > ' + cmd);
  console.log('===============================================================');
  let complete = false;
  const childProcess = spawn(cmd, { shell: true, stdio: 'inherit', cwd: TEST_E2E_RUN_FOLDER });
  childProcess.on('close', function (code) {
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
  return new Promise(function (resolve) {
    setTimeout(function () {
      resolve();
    }, ms);
  });
}

main();
