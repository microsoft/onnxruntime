// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const assert = require('node:assert/strict');
const path = require('node:path');

// The native binding accepts a Tensor constructor independently of the TypeScript package.
class Tensor {
  constructor(type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = dims;
    this.location = 'cpu';
  }
}

async function main() {
  const binding = require(path.resolve(process.argv[2]));
  binding.initOrtOnce(2, Tensor, true);
  const session = new binding.InferenceSession();
  await session.loadModel(path.resolve(process.argv[3]), {intraOpNumThreads: 1});
  try {
    const a = new Tensor('float32', new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const b = new Tensor('float32', new Float32Array([10, 20, 30, 40, 50, 60]), [2, 3]);
    const output = session.run({A: a, B: b}, {C: null}, {}).C;
    assert.deepEqual(output.dims, [2, 3]);
    assert.deepEqual(Array.from(output.data), [11, 22, 33, 44, 55, 66]);
    assert.equal(output.type, 'float32');
    console.log('Node-API Add inference passed.');
  } finally {
    session.dispose();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
