// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { hasPackageVersion, verifyFileIntegrity } = require('./install-utils.js');

const withTempFile = async (contents, fn) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'onnxruntime-node-integrity-'));
  const filepath = path.join(directory, 'package.nupkg');
  fs.writeFileSync(filepath, contents);
  try {
    await fn(filepath);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
};

test('accepts a matching package digest', async () => {
  const contents = Buffer.from('package contents');
  const integrity = `sha512-${crypto.createHash('sha512').update(contents).digest('base64')}`;
  await withTempFile(contents, async (filepath) => verifyFileIntegrity(filepath, integrity));
});

test('rejects modified package contents', async () => {
  const expectedContents = Buffer.from('package contents');
  const integrity = `sha512-${crypto.createHash('sha512').update(expectedContents).digest('base64')}`;
  await withTempFile(Buffer.from('modified contents'), async (filepath) => {
    await assert.rejects(verifyFileIntegrity(filepath, integrity), /integrity check failed/);
  });
});

test('rejects missing or malformed integrity metadata', async () => {
  await withTempFile(Buffer.from('package contents'), async (filepath) => {
    await assert.rejects(verifyFileIntegrity(filepath, undefined), /Missing or invalid/);
    await assert.rejects(verifyFileIntegrity(filepath, 'sha256-not-supported'), /Missing or invalid/);
    await assert.rejects(verifyFileIntegrity(filepath, 'sha512-YQ=='), /Missing or invalid/);
  });
});

test('treats unavailable package versions as fallback candidates', () => {
  assert.equal(hasPackageVersion(null, '1.32.0'), false);
  assert.equal(hasPackageVersion({}, '1.32.0'), false);
  assert.equal(hasPackageVersion({ versions: ['1.31.0'] }, '1.32.0'), false);
  assert.equal(hasPackageVersion({ versions: ['1.31.0', '1.32.0'] }, '1.32.0'), true);
});
