// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const assert = require('node:assert/strict');
const path = require('node:path');
const {spawnSync} = require('node:child_process');

const files = process.argv.slice(2);
const loader = files.find((file) => file.endsWith('.js'));
assert(loader, 'Missing Bazel-built WASM test loader; pass the test module outputs.');
const filter = process.env.TESTBRIDGE_TEST_ONLY || '*';
assert(!filter.startsWith('-'), 'Select a positive gtest case with --test_filter=WebAssemblyTest.test.');
const result = spawnSync(process.execPath, [path.resolve(loader), `--gtest_filter=${filter}`], {
  cwd: path.dirname(path.resolve(loader)),
  encoding: 'utf8',
});
if (result.error) {
  throw result.error;
}
process.stdout.write(result.stdout);
process.stderr.write(result.stderr);
assert.equal(result.status, 0, `WASM test failed (status ${result.status}, signal ${result.signal}).`);
assert.match(result.stdout, /\[  PASSED  \] [1-9]\d* test/,
    `No WASM tests passed for --test_filter=${filter}; select an existing WebAssemblyTest case.`);
