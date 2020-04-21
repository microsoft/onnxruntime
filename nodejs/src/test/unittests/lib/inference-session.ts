import assert from 'assert';
import * as path from 'path';

import {Tensor, TypedTensor} from '../../../lib';
import {InferenceSession} from '../../../lib/inference-session';
import {assertDataEqual, assertTensorEqual} from '../../test-utils';

const MODEL_TEST_TYPES_CASES:
    {model: string, type: Tensor.Type, input0: Tensor.DataType, expectedOutput0: Tensor.DataType}[] = [
      {
        model: path.join(__dirname, '../../testdata/test_types_BOOL.pb'),
        type: 'bool',
        input0: Uint8Array.from([1, 0, 0, 1, 0]),
        expectedOutput0: Uint8Array.from([1, 0, 0, 1, 0])
      },
      {
        model: path.join(__dirname, '../../testdata/test_types_DOUBLE.pb'),
        type: 'float64',
        input0: Float64Array.from([1.0, 2.0, 3.0, 4.0, 5.0]),
        expectedOutput0: Float64Array.from([1.0, 2.0, 3.0, 4.0, 5.0])
      },
      {
        model: path.join(__dirname, '../../testdata/test_types_FLOAT.pb'),
        type: 'float32',
        input0: Float32Array.from([1.0, 2.0, 3.0, 4.0, 5.0]),
        expectedOutput0: Float32Array.from([1.0, 2.0, 3.0, 4.0, 5.0])
      },
      {
        model: path.join(__dirname, '../../testdata/test_types_INT8.pb'),
        type: 'int8',
        input0: Int8Array.from([1, -2, 3, 4, -5]),
        expectedOutput0: Int8Array.from([1, -2, 3, 4, -5])
      },
      {
        model: path.join(__dirname, '../../testdata/test_types_INT16.pb'),
        type: 'int16',
        input0: Int16Array.from([1, -2, 3, 4, -5]),
        expectedOutput0: Int16Array.from([1, -2, 3, 4, -5])
      },
      {
        model: path.join(__dirname, '../../testdata/test_types_INT32.pb'),
        type: 'int32',
        input0: Int32Array.from([1, -2, 3, 4, -5]),
        expectedOutput0: Int32Array.from([1, -2, 3, 4, -5])
      },
      {
        model: path.join(__dirname, '../../testdata/test_types_INT64.pb'),
        type: 'int64',
        input0: BigInt64Array.from([BigInt(1), BigInt(-2), BigInt(3), BigInt(4), BigInt(-5)]),
        expectedOutput0: BigInt64Array.from([BigInt(1), BigInt(-2), BigInt(3), BigInt(4), BigInt(-5)])
      },
      {
        model: path.join(__dirname, '../../testdata/test_types_STRING.pb'),
        type: 'string',
        input0: ['a', 'b', 'c', 'd', 'e'],
        expectedOutput0: ['a', 'b', 'c', 'd', 'e']
      },
      {
        model: path.join(__dirname, '../../testdata/test_types_UINT8.pb'),
        type: 'uint8',
        input0: Uint8Array.from([1, 2, 3, 4, 5]),
        expectedOutput0: Uint8Array.from([1, 2, 3, 4, 5])
      },
      {
        model: path.join(__dirname, '../../testdata/test_types_UINT16.pb'),
        type: 'uint16',
        input0: Uint16Array.from([1, 2, 3, 4, 5]),
        expectedOutput0: Uint16Array.from([1, 2, 3, 4, 5])
      },
      {
        model: path.join(__dirname, '../../testdata/test_types_UINT32.pb'),
        type: 'uint32',
        input0: Uint32Array.from([1, 2, 3, 4, 5]),
        expectedOutput0: Uint32Array.from([1, 2, 3, 4, 5])
      },
      {
        model: path.join(__dirname, '../../testdata/test_types_UINT64.pb'),
        type: 'uint64',
        input0: BigUint64Array.from([BigInt(1), BigInt(2), BigInt(3), BigInt(4), BigInt(5)]),
        expectedOutput0: BigUint64Array.from([BigInt(1), BigInt(2), BigInt(3), BigInt(4), BigInt(5)])
      },
    ];

const SQUEEZENET_INPUT0_DATA = require(path.join(__dirname, '../../testdata/squeezenet.input0.json'));
const SQUEEZENET_OUTPUT0_DATA = require(path.join(__dirname, '../../testdata/squeezenet.output0.json'));

describe('UnitTests - simple E2E tests', () => {
  MODEL_TEST_TYPES_CASES.forEach(testCase => {
    it(`${testCase.model}`, async () => {
      const session = await InferenceSession.create(testCase.model);
      const output = await session.run({'input': new Tensor(testCase.type, testCase.input0, [1, 5])});
      assert(output.hasOwnProperty('output'), `'output' should be in the result object.`);
      assert(output['output'] instanceof Tensor, `result[output] should be a Tensor object.`);
      assert.strictEqual(output['output'].size, 5, `output size expected 5, got ${output['output'].size}.`);
      assert.strictEqual(
          output['output'].type, testCase.type, `tensor type expected ${testCase.type}, got ${output['output'].type}.`);
      assert.strictEqual(
          Object.getPrototypeOf(output['output'].data), Object.getPrototypeOf(testCase.expectedOutput0),
          `tensor data expected ${Object.getPrototypeOf(testCase.expectedOutput0).constructor.name}, got ${
              Object.getPrototypeOf(output['output'].data).constructor.name}`);
      assertDataEqual(testCase.type, output['output'].data, testCase.expectedOutput0);
    });
  });
});

describe('UnitTests - run', async () => {
  let session: InferenceSession|null = null;
  let sessionAny: any;
  const input0 = new Tensor('float32', SQUEEZENET_INPUT0_DATA, [1, 3, 224, 224]);
  const expectedOutput0 = new Tensor('float32', SQUEEZENET_OUTPUT0_DATA, [1, 1000, 1, 1]);

  before(async () => {
    session = await InferenceSession.create(path.join(__dirname, '../../testdata/squeezenet.onnx'));
    sessionAny = session;
  });

  it('BAD CALL - input type mismatch (null)', () => {
    assert.rejects(async () => {
      await sessionAny.run(null);
    }, TypeError);
  });
  it('BAD CALL - input type mismatch (single tensor)', () => {
    assert.rejects(async () => {
      await sessionAny.run(input0);
    }, TypeError);
  });
  it('BAD CALL - input type mismatch (tensor array)', () => {
    assert.rejects(async () => {
      await sessionAny.run([input0]);
    }, TypeError);
  });
  it('BAD CALL - input name missing', () => {
    assert.rejects(async () => {
      await sessionAny.run({});
    }, TypeError);
  });
  it('BAD CALL - input name incorrect', () => {
    assert.rejects(async () => {
      await sessionAny.run({'data_1': input0});  // correct name should be 'data_0'
    }, TypeError);
  });

  it('run() - no fetches', async () => {
    const result = await session!.run({'data_0': input0});
    assertTensorEqual(result['softmaxout_1'], expectedOutput0);
  });
  it('run() - fetches names', async () => {
    const result = await session!.run({'data_0': input0}, ['softmaxout_1']);
    assertTensorEqual(result['softmaxout_1'], expectedOutput0);
  });
  it('run() - fetches object', async () => {
    const result = await session!.run({'data_0': input0}, {'softmaxout_1': null});
    assertTensorEqual(result['softmaxout_1'], expectedOutput0);
  });
  // TODO: enable after buffer reuse is implemented
  it.skip('run() - fetches object (pre-allocated)', async () => {
    const preAllocatedOutputBuffer = new Float32Array(expectedOutput0.size);
    const result = await session!.run(
        {'data_0': input0}, {'softmaxout_1': new Tensor(preAllocatedOutputBuffer, expectedOutput0.dims)});
    const softmaxout_1 = result['softmaxout_1'] as TypedTensor<'float32'>;
    assert.strictEqual(softmaxout_1.data.buffer, preAllocatedOutputBuffer.buffer)
    assert.strictEqual(softmaxout_1.data.byteOffset, preAllocatedOutputBuffer.byteOffset)
    assertTensorEqual(result['softmaxout_1'], expectedOutput0);
  });

  // it('BAD CALL - output value type mismatch (tensor)', () => {
  //   assert.rejects(async () => {
  //     await sessionAny.run(
  //         {'data_0': input0},
  //         {'softmaxout_1': new Tensor(new Float32Array(expectedOutput0.size), expectedOutput0.dims)});
  //   }, TypeError);
  // });
  // it('BAD CALL - output value type mismatch (tensor array)', () => {
  //   assert.rejects(async () => {
  //     await sessionAny.run(
  //         {'data_0': input0},
  //         {'softmaxout_1': [new Tensor(new Float32Array(expectedOutput0.size), expectedOutput0.dims)]});
  //   }, TypeError);
  // });
});
