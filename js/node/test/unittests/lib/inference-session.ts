// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'assert';
import * as fs from 'fs';
import {InferenceSession, Tensor, TypedTensor} from 'onnxruntime-common';
import * as path from 'path';

import {assertTensorEqual} from '../../test-utils';

const SQUEEZENET_INPUT0_DATA = require(path.join(__dirname, '../../testdata/squeezenet.input0.json'));
const SQUEEZENET_OUTPUT0_DATA = require(path.join(__dirname, '../../testdata/squeezenet.output0.json'));

describe('UnitTests - InferenceSession.create()', () => {
  const modelPath = path.join(__dirname, '../../testdata/squeezenet.onnx');
  const modelBuffer = fs.readFileSync(modelPath);
  const createAny: any = InferenceSession.create;

  //#region test bad arguments
  it('BAD CALL - no argument', async () => {
    await assert.rejects(async () => {
      await createAny();
    }, {name: 'TypeError', message: /argument\[0\]/});
  });
  it('BAD CALL - byteOffset negative number (ArrayBuffer, number)', async () => {
    await assert.rejects(async () => {
      await createAny(modelBuffer.buffer, -1);
    }, {name: 'RangeError', message: /'byteOffset'/});
  });
  it('BAD CALL - byteOffset out of range (ArrayBuffer, number)', async () => {
    await assert.rejects(async () => {
      await createAny(modelBuffer.buffer, 100000000);
    }, {name: 'RangeError', message: /'byteOffset'/});
  });
  it('BAD CALL - byteLength negative number (ArrayBuffer, number)', async () => {
    await assert.rejects(async () => {
      await createAny(modelBuffer.buffer, 0, -1);
    }, {name: 'RangeError', message: /'byteLength'/});
  });
  it('BAD CALL - byteLength out of range (ArrayBuffer, number)', async () => {
    await assert.rejects(async () => {
      await createAny(modelBuffer.buffer, 0, 100000000);
    }, {name: 'RangeError', message: /'byteLength'/});
  });
  it('BAD CALL - options type mismatch (string, string)', async () => {
    await assert.rejects(async () => {
      await createAny(modelPath, 'cpu');
    }, {name: 'TypeError', message: /'options'/});
  });
  it('BAD CALL - options type mismatch (Uint8Array, string)', async () => {
    await assert.rejects(async () => {
      await createAny(modelBuffer, 'cpu');
    }, {name: 'TypeError', message: /'options'/});
  });
  it('BAD CALL - options type mismatch (ArrayBuffer, number, number, string)', async () => {
    await assert.rejects(async () => {
      await createAny(modelBuffer.buffer, modelBuffer.byteOffset, modelBuffer.byteLength, 'cpu');
    }, {name: 'TypeError', message: /'options'/});
  });

  it('EXPECTED FAILURE - Load model failed', async () => {
    await assert.rejects(async () => {
      await InferenceSession.create('/this/is/an/invalid/path.onnx');
    }, {name: 'Error', message: /failed/});
  });
  it('EXPECTED FAILURE - empty buffer', async () => {
    await assert.rejects(async () => {
      await InferenceSession.create(new Uint8Array(0));
    }, {name: 'Error', message: /No graph was found in the protobuf/});
  });
  //#endregion

  it('metadata: inputNames', async () => {
    const session = await InferenceSession.create(modelPath);
    assert.deepStrictEqual(session.inputNames, ['data_0']);
  });
  it('metadata: outputNames', async () => {
    const session = await InferenceSession.create(modelPath);
    assert.deepStrictEqual(session.outputNames, ['softmaxout_1']);
  });
});

describe('UnitTests - InferenceSession.run()', () => {
  let session: InferenceSession|null = null;
  let sessionAny: any;
  const input0 = new Tensor('float32', SQUEEZENET_INPUT0_DATA, [1, 3, 224, 224]);
  const expectedOutput0 = new Tensor('float32', SQUEEZENET_OUTPUT0_DATA, [1, 1000, 1, 1]);

  before(async () => {
    session = await InferenceSession.create(path.join(__dirname, '../../testdata/squeezenet.onnx'));
    sessionAny = session;
  });

  //#region test bad input(feeds)
  it('BAD CALL - input type mismatch (null)', async () => {
    await assert.rejects(async () => {
      await sessionAny.run(null);
    }, {name: 'TypeError', message: /'feeds'/});
  });
  it('BAD CALL - input type mismatch (single tensor)', async () => {
    await assert.rejects(async () => {
      await sessionAny.run(input0);
    }, {name: 'TypeError', message: /'feeds'/});
  });
  it('BAD CALL - input type mismatch (tensor array)', async () => {
    await assert.rejects(async () => {
      await sessionAny.run([input0]);
    }, {name: 'TypeError', message: /'feeds'/});
  });
  it('EXPECTED FAILURE - input name missing', async () => {
    await assert.rejects(async () => {
      await sessionAny.run({});
    }, {name: 'Error', message: /input 'data_0' is missing/});
  });
  it('EXPECTED FAILURE - input name incorrect', async () => {
    await assert.rejects(async () => {
      await sessionAny.run({'data_1': input0});  // correct name should be 'data_0'
    }, {name: 'Error', message: /input 'data_0' is missing/});
  });
  //#endregion

  //#region test fetches overrides
  it('run() - no fetches', async () => {
    const result = await session!.run({'data_0': input0});
    assertTensorEqual(result.softmaxout_1, expectedOutput0);
  });
  it('run() - fetches names', async () => {
    const result = await session!.run({'data_0': input0}, ['softmaxout_1']);
    assertTensorEqual(result.softmaxout_1, expectedOutput0);
  });
  it('run() - fetches object', async () => {
    const result = await session!.run({'data_0': input0}, {'softmaxout_1': null});
    assertTensorEqual(result.softmaxout_1, expectedOutput0);
  });
  // TODO: enable after buffer reuse is implemented
  it.skip('run() - fetches object (pre-allocated)', async () => {
    const preAllocatedOutputBuffer = new Float32Array(expectedOutput0.size);
    const result = await session!.run(
        {'data_0': input0}, {'softmaxout_1': new Tensor(preAllocatedOutputBuffer, expectedOutput0.dims)});
    const softmaxout_1 = result.softmaxout_1 as TypedTensor<'float32'>;
    assert.strictEqual(softmaxout_1.data.buffer, preAllocatedOutputBuffer.buffer);
    assert.strictEqual(softmaxout_1.data.byteOffset, preAllocatedOutputBuffer.byteOffset);
    assertTensorEqual(result.softmaxout_1, expectedOutput0);
  });
  //#endregion

  //#region test bad output(fetches)
  it('BAD CALL - fetches type mismatch (null)', async () => {
    await assert.rejects(async () => {
      await sessionAny.run({'data_0': input0}, null);
    }, {name: 'TypeError', message: /argument\[1\]/});
  });
  it('BAD CALL - fetches type mismatch (number)', async () => {
    await assert.rejects(async () => {
      await sessionAny.run({'data_0': input0}, 1);
    }, {name: 'TypeError', message: /argument\[1\]/});
  });
  it('BAD CALL - fetches type mismatch (Tensor)', async () => {
    await assert.rejects(async () => {
      await sessionAny.run(
          {'data_0': input0}, new Tensor(new Float32Array(expectedOutput0.size), expectedOutput0.dims));
    }, {name: 'TypeError', message: /'fetches'/});
  });
  it('BAD CALL - fetches as array (empty array)', async () => {
    await assert.rejects(async () => {
      await sessionAny.run({'data_0': input0}, []);
    }, {name: 'TypeError', message: /'fetches'/});
  });
  it('BAD CALL - fetches as array (non-string elements)', async () => {
    await assert.rejects(async () => {
      await sessionAny.run({'data_0': input0}, [1, 2, 3]);
    }, {name: 'TypeError', message: /'fetches'/});
  });
  it('BAD CALL - fetches as array (invalid name)', async () => {
    await assert.rejects(async () => {
      await sessionAny.run({'data_0': input0}, ['im_a_wrong_output_name']);
    }, {name: 'RangeError', message: /'fetches'/});
  });
  //#endregion

  it('BAD CALL - options type mismatch (number)', async () => {
    await assert.rejects(async () => {
      await sessionAny.run({'data_0': input0}, ['softmaxout_1'], 1);
    }, {name: 'TypeError', message: /'options'/});
  });
});

describe('UnitTests - InferenceSession.SessionOptions', () => {
  const modelPath = path.join(__dirname, '../../testdata/test_types_FLOAT.pb');
  const createAny: any = InferenceSession.create;

  it('BAD CALL - type mismatch', async () => {
    await assert.rejects(async () => {
      await createAny(modelPath, 'cpu');
    }, {name: 'TypeError', message: /'options'/});
  });

  describe('executionProviders', () => {
    it.skip('BAD CALL - type mismatch', async () => {
      await assert.rejects(async () => {
        await createAny(modelPath, {executionProviders: 'bad-EP-name'});
      }, {name: 'TypeError', message: /executionProviders/});
    });
    it.skip('EXPECTED FAILURE - invalid EP name, string list', async () => {
      await assert.rejects(async () => {
        await createAny(modelPath, {executionProviders: ['bad-EP-name']});
      }, {name: 'Error', message: /executionProviders.+bad-EP-name/});
    });
    it.skip('EXPECTED FAILURE - invalid EP name, object list', async () => {
      await assert.rejects(async () => {
        await createAny(modelPath, {executionProviders: [{name: 'bad-EP-name'}]});
      }, {name: 'Error', message: /executionProviders.+bad-EP-name/});
    });
    it('string list (CPU)', async () => {
      await InferenceSession.create(modelPath, {executionProviders: ['cpu']});
    });
    it('object list (CPU)', async () => {
      await InferenceSession.create(modelPath, {executionProviders: [{name: 'cpu'}]});
    });
  });

  describe('intraOpNumThreads', () => {
    it('BAD CALL - type mismatch', async () => {
      await assert.rejects(async () => {
        await createAny(modelPath, {intraOpNumThreads: 'bad-value'});
      }, {name: 'TypeError', message: /intraOpNumThreads/});
    });
    it('BAD CALL - non-integer', async () => {
      await assert.rejects(async () => {
        await createAny(modelPath, {intraOpNumThreads: 1.5});
      }, {name: 'RangeError', message: /intraOpNumThreads/});
    });
    it('BAD CALL - negative integer', async () => {
      await assert.rejects(async () => {
        await createAny(modelPath, {intraOpNumThreads: -1});
      }, {name: 'RangeError', message: /intraOpNumThreads/});
    });
    it('intraOpNumThreads = 1', async () => {
      await InferenceSession.create(modelPath, {intraOpNumThreads: 1});
    });
  });

  describe('interOpNumThreads', () => {
    it('BAD CALL - type mismatch', async () => {
      await assert.rejects(async () => {
        await createAny(modelPath, {interOpNumThreads: 'bad-value'});
      }, {name: 'TypeError', message: /interOpNumThreads/});
    });
    it('BAD CALL - non-integer', async () => {
      await assert.rejects(async () => {
        await createAny(modelPath, {interOpNumThreads: 1.5});
      }, {name: 'RangeError', message: /interOpNumThreads/});
    });
    it('BAD CALL - negative integer', async () => {
      await assert.rejects(async () => {
        await createAny(modelPath, {interOpNumThreads: -1});
      }, {name: 'RangeError', message: /interOpNumThreads/});
    });
    it('interOpNumThreads = 1', async () => {
      await InferenceSession.create(modelPath, {interOpNumThreads: 1});
    });
  });

  describe('graphOptimizationLevel', () => {
    it('BAD CALL - type mismatch', async () => {
      await assert.rejects(async () => {
        await createAny(modelPath, {graphOptimizationLevel: 0});
      }, {name: 'TypeError', message: /graphOptimizationLevel/});
    });
    it('BAD CALL - invalid config', async () => {
      await assert.rejects(async () => {
        await createAny(modelPath, {graphOptimizationLevel: 'bad-value'});
      }, {name: 'TypeError', message: /graphOptimizationLevel/});
    });
    it('graphOptimizationLevel = basic', async () => {
      await InferenceSession.create(modelPath, {graphOptimizationLevel: 'basic'});
    });
  });

  describe('enableCpuMemArena', () => {
    it('BAD CALL - type mismatch', async () => {
      await assert.rejects(async () => {
        await createAny(modelPath, {enableCpuMemArena: 0});
      }, {name: 'TypeError', message: /enableCpuMemArena/});
    });
    it('enableCpuMemArena = true', async () => {
      await InferenceSession.create(modelPath, {enableCpuMemArena: true});
    });
  });

  describe('enableMemPattern', () => {
    it('BAD CALL - type mismatch', async () => {
      await assert.rejects(async () => {
        await createAny(modelPath, {enableMemPattern: 0});
      }, {name: 'TypeError', message: /enableMemPattern/});
    });
    it('enableMemPattern = true', async () => {
      await InferenceSession.create(modelPath, {enableMemPattern: true});
    });
  });

  describe('executionMode', () => {
    it('BAD CALL - type mismatch', async () => {
      await assert.rejects(async () => {
        await createAny(modelPath, {executionMode: 0});
      }, {name: 'TypeError', message: /executionMode/});
    });
    it('BAD CALL - invalid config', async () => {
      await assert.rejects(async () => {
        await createAny(modelPath, {executionMode: 'bad-value'});
      }, {name: 'TypeError', message: /executionMode/});
    });
    it('executionMode = sequential', async () => {
      await InferenceSession.create(modelPath, {executionMode: 'sequential'});
    });
  });
});

describe('UnitTests - InferenceSession.RunOptions', () => {
  let session: InferenceSession|null = null;
  let sessionAny: any;
  const input0 = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
  const expectedOutput0 = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);

  before(async () => {
    const modelPath = path.join(__dirname, '../../testdata/test_types_FLOAT.pb');
    session = await InferenceSession.create(modelPath);
    sessionAny = session;
  });

  describe('logSeverityLevel', () => {
    it('BAD CALL - type mismatch', async () => {
      await assert.rejects(async () => {
        await sessionAny.run({input: input0}, {logSeverityLevel: 'error'});
      }, {name: 'TypeError', message: /logSeverityLevel/});
    });
    it('BAD CALL - out of range', async () => {
      await assert.rejects(async () => {
        await sessionAny.run({input: input0}, {logSeverityLevel: 8});
      }, {name: 'RangeError', message: /logSeverityLevel/});
    });
    it('BAD CALL - out of range', async () => {
      await assert.rejects(async () => {
        await sessionAny.run({input: input0}, {logSeverityLevel: 8});
      }, {name: 'RangeError', message: /logSeverityLevel/});
    });
    it('logSeverityLevel = 4', async () => {
      const result = await sessionAny.run({input: input0}, {logSeverityLevel: 4});
      assertTensorEqual(result.output, expectedOutput0);
    });
  });
});
