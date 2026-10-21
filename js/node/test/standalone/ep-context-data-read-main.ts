// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as path from 'path';
const ort = require(path.join(__dirname, '../../'));
import * as process from 'process';

// a minimal MatMul model, same as the one used by ./main.ts
const modelData =
  'CAMSDGJhY2tlbmQtdGVzdDpiChEKAWEKAWISAWMiBk1hdE11bBIOdGVzdF9tYXRtdWxfMmRaEwoBYRIOCgwIARIICgIIAwoCCARaEwoBYhIOCgwIARIICgIIBAoCCANiEwoBYxIOCgwIARIICgIIAwoCCANCAhAJ';

const shouldFailFirst = process.argv.includes('--fail-first');
const shouldRelease = process.argv.includes('--release');

// This script intentionally never calls process.exit(): it verifies that a configured
// `epContextDataRead` callback does not keep the Node.js event loop alive.
async function main() {
  try {
    const epContextDataRead = {
      callback: (name: string) => {
        throw new Error(`unexpected EPContext data request for '${name}'`);
      },
      maxDataSize: 1024 * 1024,
    };

    if (shouldFailFirst) {
      try {
        await ort.InferenceSession.create('/this/is/an/invalid/path.onnx', { epContextDataRead });
        console.error('ERROR: expected the session creation to fail');
        process.exit(1);
      } catch {
        console.log('SUCCESS: Session creation failed as expected');
      }
    }

    const modelBuffer = Buffer.from(modelData, 'base64');
    const session = await ort.InferenceSession.create(modelBuffer, { epContextDataRead });

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
    }
  } catch (e) {
    console.error(`ERROR: ${e}`);
    process.exit(1);
  }
}

void main();
