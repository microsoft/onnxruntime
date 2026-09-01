// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'assert';
import { InferenceSession, Tensor } from 'onnxruntime-common';
import * as path from 'path';

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

  it('does not release the session or input tensor while a run is pending', async () => {
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'));
    const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
    const boundDispose = input.dispose.bind(input);
    const prototypeDispose = (Object.getPrototypeOf(input) as { dispose: () => void }).dispose;
    const run = localSession.run({ input });

    assert.throws(() => input.dispose(), /asynchronous inference/);
    assert.throws(boundDispose, /asynchronous inference/);
    assert.throws(() => prototypeDispose.call(input), /asynchronous inference/);
    assert.throws(() => input.getData(), /asynchronous inference/);
    await assert.rejects(localSession.release(), /Cannot dispose session while inference is running/);

    await run;
    input.dispose();
    await localSession.release();
  });

  it('keeps a Tensor protected until overlapping runs complete', async () => {
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'));
    for (let i = 0; i < 10; ++i) {
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      const firstRun = localSession.run({ input });
      const secondRun = localSession.run({ input });

      assert.throws(() => input.dispose(), /asynchronous inference/);
      await Promise.all([firstRun, secondRun]);
      input.dispose();
    }
    await localSession.release();
  });
});
