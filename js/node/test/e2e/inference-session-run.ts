// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession} from 'onnxruntime-common';
import * as path from 'path';

import {Tensor} from '../../lib';
import {SQUEEZENET_INPUT0_DATA, SQUEEZENET_OUTPUT0_DATA, TEST_DATA_ROOT} from '../test-utils';
import {assertTensorEqual} from '../test-utils';

describe('E2E Tests - InferenceSession.run()', async () => {
  let session: InferenceSession;
  const input0 = new Tensor('float32', SQUEEZENET_INPUT0_DATA, [1, 3, 224, 224]);
  const expectedOutput0 = new Tensor('float32', SQUEEZENET_OUTPUT0_DATA, [1, 1000, 1, 1]);

  before(async () => {
    session = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'squeezenet.onnx'));
  });

  it('multiple run() calls', async () => {
    for (let i = 0; i < 1000; i++) {
      const result = await session!.run({'data_0': input0}, ['softmaxout_1']);
      assertTensorEqual(result.softmaxout_1, expectedOutput0);
    }
  }).timeout('120s');
});
