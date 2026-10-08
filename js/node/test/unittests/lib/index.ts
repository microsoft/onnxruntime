// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as assert from 'assert';
import * as fs from 'fs-extra';
import * as os from 'os';
import * as path from 'path';

import { readJsoncFileSync } from '../../test-utils';

// import * as OnnxRuntime from '../../../lib/index';

describe('UnitTests - test utilities', () => {
  it('throws on malformed JSONC test fixture files', () => {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ort-jsonc-'));
    const fixturePath = path.join(tempDir, 'fixture.jsonc');

    try {
      fs.writeFileSync(fixturePath, '{ "skip": true trailing }');
      assert.throws(() => readJsoncFileSync(fixturePath), /Failed to parse JSONC file/);
    } finally {
      fs.removeSync(tempDir);
    }
  });

  it('allows trailing commas in JSONC test fixture files', () => {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ort-jsonc-'));
    const fixturePath = path.join(tempDir, 'fixture.jsonc');

    try {
      fs.writeFileSync(fixturePath, '{ "skip": true, }');
      assert.deepStrictEqual(readJsoncFileSync<{ skip: boolean }>(fixturePath), { skip: true });
    } finally {
      fs.removeSync(tempDir);
    }
  });
});
