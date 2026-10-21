// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as path from 'path';
import { isMainThread } from 'worker_threads';
import { Tensor } from 'onnxruntime-common';
const ort = require(path.join(__dirname, '../../'));
import * as process from 'process';

const modelData =
  'CAMSDGJhY2tlbmQtdGVzdDpiChEKAWEKAWISAWMiBk1hdE11bBIOdGVzdF9tYXRtdWxfMmRaEwoBYRIOCgwIARIICgIIAwoCCARaEwoBYhIOCgwIARIICgIIBAoCCANiEwoBYxIOCgwIARIICgIIAwoCCANCAhAJ';
const shouldProcessExit = process.argv.includes('--process-exit');
const shouldThrowException = process.argv.includes('--throw-exception');
const shouldRelease = process.argv.includes('--release');
const shouldInitializeOrtTwice = process.argv.includes('--initialize-twice');

async function main() {
  try {
    if (shouldInitializeOrtTwice) {
      const binding = require(
        path.join(__dirname, `../../bin/napi-v6/${process.platform}/${process.arch}/onnxruntime_binding.node`),
      );
      let lastTensorConstructor = '';
      const createTensorConstructor = (name: string) =>
        function (type: Tensor.Type, data: Tensor.DataType, dims?: readonly number[]) {
          lastTensorConstructor = name;
          return new Tensor(type, data, dims);
        } as unknown as typeof Tensor;
      const FirstTensor = createTensorConstructor('first');
      const SecondTensor = createTensorConstructor('second');
      binding.initOrtOnce(2, FirstTensor, isMainThread);
      binding.initOrtOnce(2, SecondTensor, isMainThread);

      const modelBuffer = Buffer.from(modelData, 'base64');
      const session = new binding.InferenceSession();
      session.loadModel(modelBuffer.buffer, modelBuffer.byteOffset, modelBuffer.byteLength, {});
      const result = session.run(
        {
          a: new Tensor('float32', Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), [3, 4]),
          b: new Tensor('float32', Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]), [4, 3]),
        },
        { c: null },
        {},
      );
      if (lastTensorConstructor !== 'second' || !(result.c instanceof Tensor)) {
        throw new Error('Repeated initialization did not update the Tensor constructor.');
      }
      session.dispose();
      console.log('SUCCESS: ORT initialized twice');
      return;
    }

    const modelBuffer = Buffer.from(modelData, 'base64');
    const session = await ort.InferenceSession.create(modelBuffer);

    const dataA = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const dataB = Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);
    const tensorA = new ort.Tensor('float32', dataA, [3, 4]);
    const tensorB = new ort.Tensor('float32', dataB, [4, 3]);

    const results = await session.run({ a: tensorA, b: tensorB });
    console.log('SUCCESS: Inference completed');
    console.log(`Result: ${results.c.data}`);

    if (shouldRelease) {
      await session.release();
      console.log('Session released');
    } else {
      console.log('Session NOT released (testing cleanup behavior)');
    }

    if (shouldThrowException) {
      setTimeout(() => {
        throw new Error('Test exception');
      }, 10);
      return;
    }

    if (shouldProcessExit) {
      process.exit(0);
    }
  } catch (e) {
    console.error(`ERROR: ${e}`);
    process.exit(1);
  }
}

void main();
