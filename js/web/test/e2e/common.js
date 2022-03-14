// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

function assert(cond) {
  if (!cond) throw new Error();
}

var createSession = function(ort, options) {
  return ort.InferenceSession.create('./model.onnx', options || {});
}

var testFunction = async function (ort, options) {
  const session = await createSession(ort, options);

  const dataA = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  const dataB = Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);

  const fetches = await session.run({
    a: new ort.Tensor('float32', dataA, [3, 4]),
    b: new ort.Tensor('float32', dataB, [4, 3])
  });

  const c = fetches.c;

  assert(c instanceof ort.Tensor);
  assert(c.dims.length === 2 && c.dims[0] === 3 && c.dims[1] === 3);
  assert(c.data[0] === 700);
  assert(c.data[1] === 800);
  assert(c.data[2] === 900);
  assert(c.data[3] === 1580);
  assert(c.data[4] === 1840);
  assert(c.data[5] === 2100);
  assert(c.data[6] === 2460);
  assert(c.data[7] === 2880);
  assert(c.data[8] === 3300);
};

if (typeof module === 'object') {
  module.exports = testFunction;
}
