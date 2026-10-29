// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'assert';
import {InferenceSession,
        Tensor} from 'onnxruntime-common';
import * as path from 'path';

import {assertTensorEqual,
        SQUEEZENET_INPUT0_DATA,
        SQUEEZENET_OUTPUT0_DATA,
        TEST_DATA_ROOT} from '../test-utils';

describe('API Tests - InferenceSession.run()', async () => {
  let session: InferenceSession;
  const input0 = new Tensor('float32', SQUEEZENET_INPUT0_DATA, [1, 3, 224, 224]);
  const expectedOutput0 = new Tensor('float32', SQUEEZENET_OUTPUT0_DATA, [1, 1000, 1, 1]);

  before(async () => {
    session = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'squeezenet.onnx'));
  });

  it('multiple run() calls', async () => {
    for (let i = 0; i < 1000; i++) {
      const result = await session!.run({data_0: input0}, ['softmaxout_1']);
      assertTensorEqual(result.softmaxout_1, expectedOutput0);
    }
  }).timeout(process.arch === 'x64' ? '120s' : 0);

  const createWebGpuSession = async(context: Mocha.Context): Promise<InferenceSession> => {
    try {
      return await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'), {
        executionProviders: ['webgpu'],
        preferredOutputLocation: 'gpu-buffer',
      });
    } catch {
      return context.skip();
    }
  };

  it('keeps GPU outputs alive after releasing their session', async function() {
    const webGpuSession = await createWebGpuSession(this);
    const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
    const first = (await webGpuSession.run({input})).output;
    const second = (await webGpuSession.run({input})).output;

    await webGpuSession.release();
    await assert.rejects(first.getData(), /GPU tensor download is not implemented/);
    first.dispose();
    first.dispose();
    second.dispose();
  });

  it('returns a caller-supplied GPU output tensor', async function() {
    const ownerSession = await createWebGpuSession(this);
    const receivingSession = await createWebGpuSession(this);
    const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
    const supplied = (await ownerSession.run({input})).output;

    const result = await receivingSession.run({input}, {output: supplied});
    assert.strictEqual(result.output, supplied);

    await receivingSession.release();
    await ownerSession.release();
    supplied.dispose();
  });

  it('retains GPU input ownership for aliasing outputs', async function() {
    if (typeof global.gc !== 'function') {
      this.skip();
    }

    const producer = await createWebGpuSession(this);
    const consumer = await createWebGpuSession(this);
    const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
    let source: Tensor|undefined = (await producer.run({input})).output;
    const aliased = (await consumer.run({input: source})).output;
    assert.strictEqual(aliased.gpuBuffer, source.gpuBuffer);

    await producer.release();
    source = undefined;
    global.gc!();
    await new Promise<void>((resolve) => setImmediate(resolve));
    global.gc!();

    aliased.dispose();
    await consumer.release();
  });

  it('finalizes abandoned GPU outputs', async function() {
    if (typeof global.gc !== 'function') {
      this.skip();
    }

    const webGpuSession = await createWebGpuSession(this);
    const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
    let output: Tensor|undefined = (await webGpuSession.run({input})).output;
    assert.strictEqual(output.location, 'gpu-buffer');
    await webGpuSession.release();
    output = undefined;
    global.gc!();
    await new Promise<void>((resolve) => setImmediate(resolve));
    global.gc!();
  });
});
