// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');

function select(available, defaultFile, args, filter = '') {
  let files = [];
  let list = false;
  for (const arg of args) {
    if (arg.startsWith('--file=')) {
      const file = arg.slice('--file='.length);
      if (!available.includes(file)) {
        throw new Error(`Unknown Node test file '${file}'. Choose one of: ${available.join(', ')}.`);
      }
      files.push(file);
    } else if (arg === '--list') {
      list = true;
    } else {
      throw new Error(`Unknown Node test argument '${arg}'. Use --file=<name>, --list, or Bazel --test_filter=<regex>.`);
    }
  }
  if (!files.length) {
    files = defaultFile === 'all' ? available : [defaultFile];
  }
  let grep;
  if (filter) {
    try {
      grep = new RegExp(filter);
    } catch (error) {
      throw new SyntaxError(`Invalid Node --test_filter '${filter}': ${error.message}. ` +
          'Use a valid JavaScript regular expression; use --list to inspect case titles.', {cause: error});
    }
  }
  return {files: [...new Set(files)], grep, list};
}

function execute(mocha, list) {
  return new Promise((resolve, reject) => {
    mocha.loadFiles();
    const selected = [];
    mocha.suite.eachTest((test) => {
      if (!mocha.options.grep || mocha.options.grep.test(test.fullTitle())) {
        selected.push(test.fullTitle());
      }
    });
    if (!selected.length) {
      reject(new Error('Node test selection matched zero cases. Use --list to inspect titles and correct --test_filter.'));
      return;
    }
    if (list) {
      console.log(selected.join('\n'));
      console.log(`Listed ${selected.length} cases; no tests executed.`);
      resolve();
      return;
    }
    const runner = mocha.run((failures) => {
      if (failures || runner.stats.passes === 0) {
        reject(new Error(`Node tests: ${runner.stats.passes} passed, ${failures} failed, ${runner.stats.pending} pending. ` +
            'Inspect the Mocha failures; an all-skipped selection is not a pass.'));
      } else {
        resolve();
      }
    });
  });
}

async function main() {
  const [packagePath, addonPath, runtimePath, defaultFile, ...args] = process.argv.slice(2);
  const available = JSON.parse(fs.readFileSync(path.join(packagePath, 'test-files.json'), 'utf8'));
  const selection = select(available, defaultFile, args, process.env.TESTBRIDGE_TEST_ONLY);
  const npm = createRequire(path.resolve('bazel/node/package.json'));
  const Mocha = npm('mocha');
  const root = fs.mkdtempSync(path.join(process.env.TEST_TMPDIR, 'node-'));
  fs.cpSync(fs.realpathSync(packagePath), root, {recursive: true, dereference: true});
  // Junctions need no Windows symlink privilege; dependencies remain Bazel-managed runfiles.
  for (const name of ['fs-extra', 'jsonc', 'protobufjs']) {
    fs.symlinkSync(fs.realpathSync(path.resolve('bazel/node/node_modules', name)),
        path.join(root, 'node_modules', name), process.platform === 'win32' ? 'junction' : 'dir');
  }
  const nativeDir = path.join(root, 'js/node/bin/napi-v6', process.platform, process.arch);
  fs.mkdirSync(nativeDir, {recursive: true});
  fs.copyFileSync(addonPath, path.join(nativeDir, 'onnxruntime_binding.node'));
  fs.copyFileSync(runtimePath, path.join(nativeDir, path.basename(runtimePath)));
  require(path.join(root, 'js/node'));
  process.chdir(root);
  const mocha = new Mocha({timeout: 30000, grep: selection.grep});
  for (const file of selection.files) {
    mocha.addFile(path.join(root, 'js/node/test', `${file}.js`));
  }
  await execute(mocha, selection.list);
}

module.exports = {select, execute};
if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
