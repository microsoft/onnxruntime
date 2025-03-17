// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const path = require('path');
const fs = require('fs-extra');
const { spawn } = require('child_process');
const startServer = require('./simple-http-server');
const minimist = require('minimist');

const { NODEJS_TEST_CASES, BROWSER_TEST_CASES, BUNDLER_TEST_CASES } = require('./run-data');

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

// always use a new folder as user-data-dir
let nextUserDataDirId = 0;
function getNextUserDataDir() {
  const dir = path.resolve(CHROME_USER_DATA_FOLDER, nextUserDataDirId.toString());
  nextUserDataDirId++;
  fs.emptyDirSync(dir);
  return dir;
}

// commandline arguments
const BROWSER = minimist(process.argv.slice(2)).browser || 'Chrome_default';

async function main() {
  // find packed package
  const { globbySync } = await import('globby');

  const ORT_COMMON_FOLDER = path.resolve(JS_ROOT_FOLDER, 'common');
  const ORT_COMMON_PACKED_FILEPATH_CANDIDATES = globbySync('onnxruntime-common-*.tgz', { cwd: ORT_COMMON_FOLDER });

  const PACKAGES_TO_INSTALL = [];

  if (ORT_COMMON_PACKED_FILEPATH_CANDIDATES.length === 1) {
    PACKAGES_TO_INSTALL.push(path.resolve(ORT_COMMON_FOLDER, ORT_COMMON_PACKED_FILEPATH_CANDIDATES[0]));
  } else if (ORT_COMMON_PACKED_FILEPATH_CANDIDATES.length > 1) {
    throw new Error('multiple packages found for onnxruntime-common.');
  }

  const ORT_WEB_FOLDER = path.resolve(JS_ROOT_FOLDER, 'web');
  const ORT_WEB_PACKED_FILEPATH_CANDIDATES = globbySync('onnxruntime-web-*.tgz', { cwd: ORT_WEB_FOLDER });
  if (ORT_WEB_PACKED_FILEPATH_CANDIDATES.length !== 1) {
    throw new Error('cannot find exactly single package for onnxruntime-web.');
  }
  PACKAGES_TO_INSTALL.push(path.resolve(ORT_WEB_FOLDER, ORT_WEB_PACKED_FILEPATH_CANDIDATES[0]));

  // we start here:

  // install dev dependencies
  await runInShell(`npm install`);

  // npm install with "--cache" to install packed packages with an empty cache folder
  await runInShell(`npm install --cache "${NPM_CACHE_FOLDER}" ${PACKAGES_TO_INSTALL.map((i) => `"${i}"`).join(' ')}`);

  // prepare .wasm files for path override testing
  prepareWasmPathOverrideFiles();

  // Setup the wwwroot folder for hosting .wasm files (for cross-origin testing)
  const serverWwwRoot = path.resolve(TEST_E2E_RUN_FOLDER, 'wwwroot');
  fs.emptyDirSync(serverWwwRoot);

  // prepare ESM loaders
  prepareEsmLoaderFiles();

  await fs.symlink(
    path.resolve(TEST_E2E_RUN_FOLDER, 'node_modules', 'onnxruntime-web', 'dist'),
    path.join(serverWwwRoot, 'dist'),
    'junction',
  );
  await fs.symlink(
    path.resolve(TEST_E2E_RUN_FOLDER, 'test-wasm-path-override'),
    path.join(serverWwwRoot, 'test-wasm-path-override'),
    'junction',
  );

  // start a HTTP server for hosting .wasm files (for cross-origin testing)
  const server = startServer(serverWwwRoot, 8081);

  // await delay(1000 * 3600);  // wait for 1 hour

  try {
    // test case run in Node.js
    await testAllNodejsCases();

    // test cases with self-host (ort hosted in same origin)
    await testAllBrowserCases({ hostInKarma: true });

    // test cases without self-host (ort hosted in different origin)
    await testAllBrowserCases({ hostInKarma: false });

    // run bundlers
    await runInShell(`npm run build`);

    // test package consuming test
    await testAllBrowserPackagesConsumingCases();
  } finally {
    // close the server after all tests
    await server.close();
  }
}

function prepareEsmLoaderFiles() {
  const allEsmFiles = [...new Set(BROWSER_TEST_CASES.map((i) => i[3]).filter((i) => i && i.endsWith('.mjs')))];

  // self-hosted
  fs.emptyDirSync(path.join(TEST_E2E_RUN_FOLDER, 'esm-loaders'));
  fs.emptyDirSync(path.join(TEST_E2E_RUN_FOLDER, 'wwwroot', 'esm-loaders'));
  allEsmFiles.forEach((i) => {
    fs.writeFileSync(
      path.join(TEST_E2E_RUN_FOLDER, 'esm-loaders', i),
      `import * as x from '../node_modules/onnxruntime-web/dist/${i}'; globalThis.ort = x;`,
    );
    fs.writeFileSync(
      path.join(TEST_E2E_RUN_FOLDER, 'wwwroot', 'esm-loaders', i),
      `import * as x from '../dist/${i}'; globalThis.ort = x;`,
    );
  });
}

function prepareWasmPathOverrideFiles() {
  const folder = path.join(TEST_E2E_RUN_FOLDER, 'test-wasm-path-override');
  const sourceFile = path.join(
    TEST_E2E_RUN_FOLDER,
    'node_modules',
    'onnxruntime-web',
    'dist',
    'ort-wasm-simd-threaded',
  );
  fs.emptyDirSync(folder);
  fs.copyFileSync(`${sourceFile}.mjs`, path.join(folder, 'ort-wasm-simd-threaded.mjs'));
  fs.copyFileSync(`${sourceFile}.wasm`, path.join(folder, 'ort-wasm-simd-threaded.wasm'));
  fs.copyFileSync(`${sourceFile}.mjs`, path.join(folder, 'renamed.mjs'));
  fs.copyFileSync(`${sourceFile}.wasm`, path.join(folder, 'renamed.wasm'));
  fs.copyFileSync(`${sourceFile}.jsep.mjs`, path.join(folder, 'ort-wasm-simd-threaded.jsep.mjs'));
  fs.copyFileSync(`${sourceFile}.jsep.wasm`, path.join(folder, 'ort-wasm-simd-threaded.jsep.wasm'));
  fs.copyFileSync(`${sourceFile}.jsep.mjs`, path.join(folder, 'jsep-renamed.mjs'));
  fs.copyFileSync(`${sourceFile}.jsep.wasm`, path.join(folder, 'jsep-renamed.wasm'));
}

async function testAllNodejsCases() {
  for (const caseName of NODEJS_TEST_CASES) {
    await runInShell(`node ./node_modules/mocha/bin/mocha --timeout 10000 ${caseName}`);
  }
}

async function testAllBrowserCases({ hostInKarma }) {
  for (const [testForSameOrigin, testForCrossOrigin, main, ortMain, args] of BROWSER_TEST_CASES) {
    if (hostInKarma && testForSameOrigin) {
      await runKarma({ hostInKarma, main, ortMain, args });
      await runKarma({ hostInKarma, main, ortMain, args, enableSharedArrayBuffer: true });
    }
    if (!hostInKarma && testForCrossOrigin) {
      await runKarma({ hostInKarma, main, ortMain, args });
      await runKarma({ hostInKarma, main, ortMain, args, enableSharedArrayBuffer: true });
    }
  }
}

async function testAllBrowserPackagesConsumingCases() {
  for (const [main, format] of BUNDLER_TEST_CASES) {
    await runKarma({ hostInKarma: true, main, ortMain: '', format });
    await runKarma({ hostInKarma: true, main, ortMain: '', format, enableSharedArrayBuffer: true });
  }
}

async function runKarma({
  hostInKarma,
  main,
  browser = BROWSER,
  ortMain = 'ort.min.js',
  format = 'iife',
  enableSharedArrayBuffer = false,
  args = [],
}) {
  const selfHostFlag = hostInKarma ? '--self-host' : '';
  const argsStr = args.map((i) => `--test-args=${i}`).join(' ');
  const formatFlag = `--format=${format}`;
  const enableSharedArrayBufferFlag = enableSharedArrayBuffer ? '--enable-shared-array-buffer' : '';
  await runInShell(
    `npx karma start --single-run --browsers ${browser} ${selfHostFlag} --ort-main=${ortMain} --test-main=${
      main
    } --user-data=${getNextUserDataDir()} ${argsStr} ${formatFlag} ${enableSharedArrayBufferFlag}`,
  );
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
