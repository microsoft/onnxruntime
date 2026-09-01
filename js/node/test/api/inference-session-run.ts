// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'assert';
import { InferenceSession, Tensor } from 'onnxruntime-common';
import * as path from 'path';

import { listSupportedBackends } from '../../lib/backend';
import { assertTensorEqual, SQUEEZENET_INPUT0_DATA, SQUEEZENET_OUTPUT0_DATA, TEST_DATA_ROOT } from '../test-utils';

describe('API Tests - InferenceSession.run()', async () => {
  let session: InferenceSession;
  const input0 = new Tensor('float32', SQUEEZENET_INPUT0_DATA, [1, 3, 224, 224]);
  const expectedOutput0 = new Tensor('float32', SQUEEZENET_OUTPUT0_DATA, [1, 1000, 1, 1]);

  before(async () => {
    session = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'squeezenet.onnx'));
  });

  it('multiple run() calls', async () => {
    for (let i = 0; i < 1000; i++) {
      const result = await session!.run({ data_0: input0 }, ['softmaxout_1']);
      assertTensorEqual(result.softmaxout_1, expectedOutput0);
    }
  }).timeout(process.arch === 'x64' ? '120s' : 0);

  it('keeps native input resources alive when Tensor disposal bypasses instance methods', async () => {
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'));
    const disposeInput = [
      (input: Tensor) => input.dispose(),
      (input: Tensor) => input.dispose.bind(input)(),
      (input: Tensor) => (Object.getPrototypeOf(input) as { dispose: () => void }).dispose.call(input),
    ];

    for (const dispose of disposeInput) {
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      const run = localSession.run({ input });
      dispose(input);
      await assert.rejects(localSession.release(), /Cannot dispose session while inference is running/);
      assertTensorEqual((await run).output, new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]));
    }
    await localSession.release();
  });

  it('copies CPU input data before starting an asynchronous run', async () => {
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'));
    const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
    const inputData = input.data;
    const run = localSession.run({ input });

    inputData.fill(0);
    const result = await run;
    assertTensorEqual(result.output, new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]));
    await localSession.release();
  });

  it('keeps resources alive across overlapping runs after Tensor disposal', async () => {
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'));
    for (let i = 0; i < 10; ++i) {
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      const firstRun = localSession.run({ input });
      const secondRun = localSession.run({ input });

      input.dispose();
      const results = await Promise.all([firstRun, secondRun]);
      for (const result of results) {
        assertTensorEqual(result.output, new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]));
      }
    }
    await localSession.release();
  });

  it('keeps a GPU buffer alive when Tensor disposal bypasses instance methods', async function () {
    if (!listSupportedBackends().some((backend) => backend.name === 'webgpu')) {
      // eslint-disable-next-line no-invalid-this
      this.skip();
    }

    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'), {
      executionProviders: ['webgpu'],
      preferredOutputLocation: 'gpu-buffer',
    });
    const disposeInput = [
      (input: Tensor) => input.dispose.bind(input)(),
      (input: Tensor) => (Object.getPrototypeOf(input) as { dispose: () => void }).dispose.call(input),
    ];

    for (const dispose of disposeInput) {
      const gpuInput = (await localSession.run({ input: new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]) })).output;
      assert.strictEqual(gpuInput.location, 'gpu-buffer');

      const run = localSession.run({ input: gpuInput });
      dispose(gpuInput);
      const result = await run;
      assert.strictEqual(result.output.location, 'gpu-buffer');
      result.output.dispose();
    }
    await localSession.release();
  });
});
