// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { spawn } from 'child_process';
import * as assert from 'assert';
import * as path from 'path';

describe('Standalone Process Tests', () => {
  // Helper function to run test script in a separate process
  const runTest = async (args: string[] = []): Promise<{ code: number; stdout: string; stderr: string }> =>
    new Promise((resolve, reject) => {
      // Use the compiled main.js file from the lib directory
      const testFile = path.join(__dirname, './main.js');

      const child = spawn('node', [testFile, ...args], { stdio: 'pipe' });

      let stdout = '';
      let stderr = '';

      child.stdout.on('data', (data) => (stdout += data.toString()));
      child.stderr.on('data', (data) => (stderr += data.toString()));

      child.on('close', (code) => {
        resolve({ code: code || 0, stdout, stderr });
      });

      child.on('error', reject);
    });

  // Helper function to check basic success criteria
  const assertSuccess = (result: { code: number; stdout: string; stderr: string }) => {
    assert.strictEqual(result.code, 0);
    assert.ok(result.stdout.includes('SUCCESS: Inference completed'));
    assert.ok(!result.stderr.includes('mutex lock failed'));
  };

  // Helper function to check that no mutex crashes occurred
  const assertNoMutexErrors = (stderr: string) => {
    assert.ok(!stderr.includes('mutex lock failed'));
    assert.ok(!stderr.includes('std::__1::system_error'));
  };

  it('should handle normal process exit', async () => {
    const result = await runTest([]);
    assertSuccess(result);
  });

  it('should handle process.exit() call', async () => {
    const result = await runTest(['--process-exit']);
    assertSuccess(result);
  });

  it('should handle uncaught exceptions', async () => {
    const result = await runTest(['--throw-exception']);

    assert.notStrictEqual(result.code, 0);
    assert.ok(result.stdout.includes('SUCCESS: Inference completed'));
    assert.ok(result.stderr.includes('Test exception'));
    assertNoMutexErrors(result.stderr);
  });

  it('should handle multiple process exits consistently', async () => {
    for (let i = 0; i < 3; i++) {
      const result = await runTest(['--process-exit']);
      assertSuccess(result);
    }
  });

  it('should handle session.release() before normal exit', async () => {
    const result = await runTest(['--release']);
    assertSuccess(result);
    assert.ok(result.stdout.includes('Session released'));
  });

  it('should handle session.release() before process.exit()', async () => {
    const result = await runTest(['--release', '--process-exit']);
    assertSuccess(result);
    assert.ok(result.stdout.includes('Session released'));
  });

  it('should handle no session.release() before process.exit()', async () => {
    const result = await runTest(['--process-exit']);
    assertSuccess(result);
    assert.ok(result.stdout.includes('Session NOT released'));
  });
});
