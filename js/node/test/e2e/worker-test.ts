// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { Worker, isMainThread, parentPort } from 'node:worker_threads';
import { InferenceSession, Tensor } from 'onnxruntime-common';
import { assertTensorEqual, SQUEEZENET_INPUT0_DATA, SQUEEZENET_OUTPUT0_DATA, TEST_DATA_ROOT } from '../test-utils';
import * as path from 'path';

if (isMainThread) {
  describe('E2E Tests - worker test', () => {
    it('should run in worker', (done) => {
      const worker = new Worker(__filename, {
        stdout: true,
        stderr: true,
      });
      worker.on('message', (msg) => {
        if (msg.result === 'success') {
          done();
        } else {
          done(new Error(`Worker failed: ${msg.error}`));
        }
      });
      worker.on('error', (err) => {
        console.error(`Worker error: ${err}`);
        done(err);
      });
    });
  });
} else {
  const workerMain = async () => {
    // require onnxruntime-node.
    require('../..');

    const input0 = new Tensor('float32', SQUEEZENET_INPUT0_DATA, [1, 3, 224, 224]);
    const expectedOutput0 = new Tensor('float32', SQUEEZENET_OUTPUT0_DATA, [1, 1000, 1, 1]);

    const session = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'squeezenet.onnx'));

    const result = await session!.run({ data_0: input0 }, ['softmaxout_1']);
    console.log('result:', result);
    assertTensorEqual(result.softmaxout_1, expectedOutput0);
  };
  workerMain().then(
    () => {
      parentPort?.postMessage({ result: 'success' });
    },
    (err) => {
      parentPort?.postMessage({ result: 'failed', error: err });
    },
  );
}
