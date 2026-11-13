// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const {test} = require('node:test');
const {select, execute} = require('./node_test_runner.cjs');
const Mocha = createRequire(path.resolve('bazel/node/package.json'))('mocha');

test('Windows launcher restores native Node entrypoint path conversion', {
  skip: process.platform !== 'win32',
}, () => {
  assert.equal(process.env.MSYS2_ARG_CONV_EXCL, '');
});

const available = ['api/simple-api-tests', 'unittests/lib/model-metadata'];
test('default file, explicit files, deduplication, and regex selection', () => {
  assert.deepEqual(select(available, available[0], []).files, [available[0]]);
  assert.deepEqual(select(available, 'all', []).files, available);
  const result = select(available, available[0], [
    `--file=${available[1]}`, `--file=${available[1]}`, '--list',
  ], 'symbol shape$');
  assert.deepEqual(result.files, [available[1]]);
  assert.equal(result.list, true);
  assert.equal(result.grep.test('metadata symbol shape'), true);
});
test('unknown files, arguments, and malformed regex fail', () => {
  assert.throws(() => select(available, 'all', ['--file=missing']), /Unknown Node test file/);
  assert.throws(() => select(available, 'all', ['--typo']), /Unknown Node test argument/);
  assert.throws(() => select(available, 'all', [], '['), SyntaxError);
});
test('zero-match, all-pending, and failed cases fail; listing does not run cases', async () => {
  const dir = fs.mkdtempSync(path.join(process.env.TEST_TMPDIR, 'runner-'));
  const file = path.join(dir, 'fixture.cjs');
  fs.writeFileSync(file, "describe('fixture', () => { it('failure', () => { throw Error('expected'); }); it.skip('pending', () => {}); });");
  function fixture(grep) {
    const mocha = new Mocha({grep: new RegExp(grep), reporter: function QuietReporter() {}});
    mocha.addFile(file);
    return mocha;
  }
  await assert.rejects(execute(fixture('absent'), false), /zero cases/);
  delete require.cache[file];
  await execute(fixture('failure'), true);
  delete require.cache[file];
  await assert.rejects(execute(fixture('pending'), false), /all-skipped/);
  delete require.cache[file];
  await assert.rejects(execute(fixture('failure'), false), /1 failed/);
});
