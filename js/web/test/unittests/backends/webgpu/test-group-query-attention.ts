// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { expect } from 'chai';
import { InferenceSession } from 'onnxruntime-common';

import { onnx } from '../../../../lib/onnxjs/ort-schema/protobuf/onnx';

const createModel = (querySeqlens: 'omitted' | 'empty' | 'present'): Uint8Array => {
  const valueInfo = (name: string, elemType: number, dims: number[]): onnx.IValueInfoProto => ({
    name,
    type: { tensorType: { elemType, shape: { dim: dims.map((dimValue) => ({ dimValue })) } } },
  });
  const inputs = [
    valueInfo('query', onnx.TensorProto.DataType.FLOAT, [1, 1, 8]),
    valueInfo('key', onnx.TensorProto.DataType.FLOAT, [1, 1, 8]),
    valueInfo('value', onnx.TensorProto.DataType.FLOAT, [1, 1, 8]),
    valueInfo('past_key', onnx.TensorProto.DataType.FLOAT, [1, 1, 1, 8]),
    valueInfo('past_value', onnx.TensorProto.DataType.FLOAT, [1, 1, 1, 8]),
    valueInfo('seqlens_k', onnx.TensorProto.DataType.INT32, [1]),
    valueInfo('total_sequence_length', onnx.TensorProto.DataType.INT32, [1]),
  ];
  const nodeInputs = inputs.map((input) => input.name!);
  if (querySeqlens !== 'omitted') {
    nodeInputs.push(...new Array<string>(9).fill(''), querySeqlens === 'present' ? 'query_seqlens' : '');
  }
  if (querySeqlens === 'present') {
    inputs.push(valueInfo('query_seqlens', onnx.TensorProto.DataType.INT32, [1]));
  }
  // Proxy mode transfers this buffer; keep protobufjs's shared allocation pool attached.
  return onnx.ModelProto.encode({
    irVersion: 9,
    opsetImport: [{ domain: 'com.microsoft', version: 1 }],
    graph: {
      name: 'jsep-query-seqlens-admission',
      input: inputs,
      output: [valueInfo('output', onnx.TensorProto.DataType.FLOAT, [1, 1, 8])],
      node: [
        {
          opType: 'GroupQueryAttention',
          domain: 'com.microsoft',
          input: nodeInputs,
          output: ['output'],
          attribute: ['num_heads', 'kv_num_heads', 'causal'].map((name) => ({
            name,
            type: onnx.AttributeProto.AttributeType.INT,
            i: 1,
          })),
        },
      ],
    },
  })
    .finish()
    .slice();
};

describe('#UnitTest# - GroupQueryAttention model serialization', () => {
  it('keeps serializing models after transferring each model buffer', () => {
    for (const querySeqlens of ['omitted', 'empty', 'present'] as const) {
      const model = createModel(querySeqlens);
      const byteLength = model.byteLength;
      const transferred = structuredClone(model, { transfer: [model.buffer] });
      expect(model.byteLength).to.equal(0);
      expect(transferred.byteLength).to.equal(byteLength);
      const decoded = onnx.ModelProto.decode(transferred);
      expect(decoded.graph!.input).to.have.lengthOf(querySeqlens === 'present' ? 8 : 7);
      expect(decoded.graph!.node![0].opType).to.equal('GroupQueryAttention');
    }
  });
});

describe('#UnitTest# - JSEP GroupQueryAttention query_seqlens admission', function () {
  before(async function (this: Mocha.Context) {
    if (!(await navigator.gpu?.requestAdapter())) {
      this.skip();
    }
  });

  const options: InferenceSession.SessionOptions = {
    executionProviders: ['webgpu'],
    graphOptimizationLevel: 'disabled',
    extra: { session: { disable_cpu_ep_fallback: '1' } },
  };

  for (const querySeqlens of ['omitted', 'empty'] as const) {
    it(`constructs a JSEP session with query_seqlens ${querySeqlens}`, async () => {
      const session = await InferenceSession.create(createModel(querySeqlens), options);
      try {
        expect(session.inputNames).to.have.lengthOf(7);
        expect(session.outputNames).to.deep.equal(['output']);
      } finally {
        await session.release();
      }
    });
  }

  it('rejects a present query_seqlens input during JSEP session construction', async () => {
    try {
      const session = await InferenceSession.create(createModel('present'), options);
      await session.release();
    } catch (error) {
      if (!(error instanceof Error)) {
        throw error;
      }
      expect(error.message).to.include('GroupQueryAttention (JS): query_seqlens (input 16) is not implemented.');
      return;
    }
    expect.fail('JSEP accepted the unsupported query_seqlens input');
  });
});
