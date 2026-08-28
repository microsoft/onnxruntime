// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'assert';
import { spawn } from 'child_process';
import * as fs from 'fs';
import { InferenceSession, Tensor } from 'onnxruntime-common';
import * as path from 'path';

import { binding } from '../../../lib/binding';
import { assertTensorEqual } from '../../test-utils';

const SQUEEZENET_INPUT0_DATA = require(path.join(__dirname, '../../testdata/squeezenet.input0.json'));
const SQUEEZENET_OUTPUT0_DATA = require(path.join(__dirname, '../../testdata/squeezenet.output0.json'));

const MODEL_PATH = path.join(__dirname, '../../testdata/squeezenet.onnx');
const SMALL_MODEL_PATH = path.join(__dirname, '../../testdata/test_types_float.onnx');

const validOptions = (): InferenceSession.SessionOptions => ({
  epContextDataRead: { callback: () => new Uint8Array(0), maxDataSize: 1024 * 1024 },
});

describe('UnitTests - InferenceSession.SessionOptions.epContextDataRead', () => {
  const createAny: any = InferenceSession.create;

  const assertTypeError = async (epContextDataRead: unknown) => {
    await assert.rejects(
      async () => {
        await createAny(SMALL_MODEL_PATH, { epContextDataRead });
      },
      { name: 'TypeError', message: /epContextDataRead/ },
    );
  };

  const assertRangeError = async (maxDataSize: unknown) => {
    await assert.rejects(
      async () => {
        await createAny(SMALL_MODEL_PATH, { epContextDataRead: { callback: () => new Uint8Array(0), maxDataSize } });
      },
      { name: 'RangeError', message: /epContextDataRead\.maxDataSize/ },
    );
  };

  // #region setup validation

  it('BAD CALL - epContextDataRead is not an object', async () => {
    await assertTypeError('yes please');
  });
  it('BAD CALL - epContextDataRead is a number', async () => {
    await assertTypeError(42);
  });
  it('BAD CALL - callback is missing', async () => {
    await assertTypeError({ maxDataSize: 1024 });
  });
  it('BAD CALL - callback is not a function', async () => {
    await assertTypeError({ callback: 'not a function', maxDataSize: 1024 });
  });
  it('BAD CALL - callback is null', async () => {
    await assertTypeError({ callback: null, maxDataSize: 1024 });
  });
  it('BAD CALL - maxDataSize is missing', async () => {
    await assertTypeError({ callback: () => new Uint8Array(0) });
  });
  it('BAD CALL - maxDataSize is not a number', async () => {
    await assertTypeError({ callback: () => new Uint8Array(0), maxDataSize: '1024' });
  });
  it('BAD CALL - maxDataSize is NaN', async () => {
    await assertRangeError(NaN);
  });
  it('BAD CALL - maxDataSize is Infinity', async () => {
    await assertRangeError(Infinity);
  });
  it('BAD CALL - maxDataSize is not an integer', async () => {
    await assertRangeError(1024.5);
  });
  it('BAD CALL - maxDataSize is zero', async () => {
    await assertRangeError(0);
  });
  it('BAD CALL - maxDataSize is negative', async () => {
    await assertRangeError(-1);
  });
  it('BAD CALL - maxDataSize is above Number.MAX_SAFE_INTEGER', async () => {
    await assertRangeError(Number.MAX_SAFE_INTEGER + 1);
  });

  it('epContextDataRead is optional', async () => {
    const session = await InferenceSession.create(SMALL_MODEL_PATH, {});
    await session.release();
  });
  it('epContextDataRead accepts undefined', async () => {
    const session = await createAny(SMALL_MODEL_PATH, { epContextDataRead: undefined });
    await session.release();
  });
  it('epContextDataRead accepts null', async () => {
    const session = await createAny(SMALL_MODEL_PATH, { epContextDataRead: null });
    await session.release();
  });
  it('epContextDataRead accepts a valid configuration', async () => {
    const session = await InferenceSession.create(SMALL_MODEL_PATH, validOptions());
    await session.release();
  });
  it('epContextDataRead accepts maxDataSize === 1', async () => {
    const session = await InferenceSession.create(SMALL_MODEL_PATH, {
      epContextDataRead: { callback: () => new Uint8Array(0), maxDataSize: 1 },
    });
    await session.release();
  });

  it('the callback is not invoked for a model without EPContext data', async function () {
    // eslint-disable-next-line no-invalid-this
    this.timeout(60000);

    let callCount = 0;
    const session = await InferenceSession.create(MODEL_PATH, {
      epContextDataRead: {
        callback: (name: string) => {
          callCount++;
          throw new Error(`unexpected callback for '${name}'`);
        },
        maxDataSize: 1024 * 1024,
      },
    });
    await session.release();
    assert.strictEqual(callCount, 0);
  });
  // #endregion

  // #region asynchronous session construction

  it('the event loop stays available while the session is being created', async function () {
    // eslint-disable-next-line no-invalid-this
    this.timeout(60000);

    let ticks = 0;
    let settled = false;
    const spin = () => {
      if (settled) {
        return;
      }
      ticks++;
      setImmediate(spin);
    };
    setImmediate(spin);

    const session = await InferenceSession.create(MODEL_PATH, validOptions());
    const ticksDuringCreate = ticks;
    settled = true;
    await session.release();

    // A synchronous native session constructor would block the event loop, so the spin counter could
    // only advance a couple of times. Genuine asynchronous construction lets it advance freely.
    assert.ok(
      ticksDuringCreate > 5,
      `expected the event loop to keep running, but it only ticked ${ticksDuringCreate} times`,
    );
  });

  it('concurrent session creation', async function () {
    // eslint-disable-next-line no-invalid-this
    this.timeout(60000);

    const sessions = await Promise.all([
      InferenceSession.create(MODEL_PATH, validOptions()),
      InferenceSession.create(MODEL_PATH, validOptions()),
      InferenceSession.create(SMALL_MODEL_PATH, validOptions()),
    ]);
    for (const session of sessions) {
      await session.release();
    }
  });
  // #endregion

  // #region model buffer lifetime

  it('the model buffer stays alive until the native session is created', async function () {
    // eslint-disable-next-line no-invalid-this
    this.timeout(60000);

    // Wrap the model bytes in a view with a non-zero byteOffset so that the offset handling is covered.
    const modelBytes = fs.readFileSync(MODEL_PATH);
    const padded = new Uint8Array(modelBytes.byteLength + 8);
    padded.set(modelBytes, 8);
    let modelData: Uint8Array | null = padded.subarray(8);

    const promise = InferenceSession.create(modelData, validOptions());
    // Drop the only JavaScript reference to the view while the native session is still being created.
    modelData = null;
    global.gc?.();

    const session = await promise;
    assert.deepStrictEqual(session.inputNames, ['data_0']);

    const output = await session.run({
      data_0: new Tensor('float32', SQUEEZENET_INPUT0_DATA, [1, 3, 224, 224]),
    });
    assertTensorEqual(output.softmaxout_1, new Tensor('float32', SQUEEZENET_OUTPUT0_DATA, [1, 1000, 1, 1]));
    await session.release();
  });

  it('the model buffer can be transferred after session creation starts', async function () {
    // eslint-disable-next-line no-invalid-this
    this.timeout(60000);

    const modelData = new Uint8Array(fs.readFileSync(MODEL_PATH));
    const session = new binding.InferenceSession();
    const promise = session.loadModel(modelData.buffer, modelData.byteOffset, modelData.byteLength, validOptions());
    structuredClone(modelData.buffer, { transfer: [modelData.buffer] });
    assert.strictEqual(modelData.byteLength, 0);

    await promise;
    assert.deepStrictEqual(
      session.inputMetadata.map(({ name }) => name),
      ['data_0'],
    );
    session.dispose();
  });

  it('EXPECTED FAILURE - empty buffer', async () => {
    await assert.rejects(
      async () => {
        await createAny(new Uint8Array(0), validOptions());
      },
      { name: 'Error', message: /Model data pointer is null./ },
    );
  });
  // #endregion

  // #region failed construction and disposal

  it('EXPECTED FAILURE - invalid model path', async () => {
    await assert.rejects(
      async () => {
        await createAny('/this/is/an/invalid/path.onnx', validOptions());
      },
      { name: 'Error', message: /failed/ },
    );
  });

  it('a failed session creation does not break the next one', async function () {
    // eslint-disable-next-line no-invalid-this
    this.timeout(60000);

    for (let i = 0; i < 3; i++) {
      await assert.rejects(async () => {
        await createAny('/this/is/an/invalid/path.onnx', validOptions());
      });
    }

    const session = await InferenceSession.create(SMALL_MODEL_PATH, validOptions());
    await session.release();
  });

  it('release() twice fails', async () => {
    const session = await InferenceSession.create(SMALL_MODEL_PATH, validOptions());
    await session.release();
    await assert.rejects(async () => {
      await session.release();
    }, /disposed/);
  });

  it('the session can be left unreleased', async () => {
    // The callback state is owned by the session wrapper and released by its finalizer.
    await InferenceSession.create(SMALL_MODEL_PATH, validOptions());
  });
  // #endregion
});

describe('UnitTests - InferenceSession.SessionOptions.epContextDataRead (standalone process)', () => {
  const runTest = async (
    args: string[] = [],
  ): Promise<{ code: number | null; signal: NodeJS.Signals | null; stdout: string; stderr: string }> =>
    new Promise((resolve, reject) => {
      const testFile = path.join(__dirname, '../../standalone/ep-context-data-read-main.js');
      const child = spawn('node', [testFile, ...args], { stdio: 'pipe' });

      let stdout = '';
      let stderr = '';

      child.stdout.on('data', (data) => (stdout += data.toString()));
      child.stderr.on('data', (data) => (stderr += data.toString()));

      child.on('close', (code, signal) => resolve({ code, signal, stdout, stderr }));
      child.on('error', reject);
    });

  // The callback must not keep the Node.js event loop alive: these processes never call process.exit(),
  // so a leaked thread-safe function reference would hang them until the test times out.
  it('the process exits after a successful session creation', async function () {
    // eslint-disable-next-line no-invalid-this
    this.timeout(60000);

    const result = await runTest();
    assert.strictEqual(result.signal, null, `process terminated by ${result.signal}`);
    assert.strictEqual(result.code, 0, result.stderr);
    assert.ok(result.stdout.includes('SUCCESS: Inference completed'), result.stdout);
  });

  it('the process exits after releasing the session', async function () {
    // eslint-disable-next-line no-invalid-this
    this.timeout(60000);

    const result = await runTest(['--release']);
    assert.strictEqual(result.signal, null, `process terminated by ${result.signal}`);
    assert.strictEqual(result.code, 0, result.stderr);
    assert.ok(result.stdout.includes('Session released'), result.stdout);
  });

  it('the process exits after a failed session creation', async function () {
    // eslint-disable-next-line no-invalid-this
    this.timeout(60000);

    const result = await runTest(['--fail-first']);
    assert.strictEqual(result.signal, null, `process terminated by ${result.signal}`);
    assert.strictEqual(result.code, 0, result.stderr);
    assert.ok(result.stdout.includes('SUCCESS: Session creation failed as expected'), result.stdout);
    assert.ok(result.stdout.includes('SUCCESS: Inference completed'), result.stdout);
  });
});
