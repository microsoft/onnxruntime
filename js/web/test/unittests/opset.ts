// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { expect } from 'chai';

import { Attribute } from '../../lib/onnxjs/attribute';
import { WEBGL_OP_RESOLVE_RULES } from '../../lib/onnxjs/backends/webgl/op-resolve-rules';
import { Graph } from '../../lib/onnxjs/graph';
import { OpSet, resolveOperator } from '../../lib/onnxjs/opset';
import { Tensor } from '../../lib/onnxjs/tensor';
import { DataType } from '../../lib/wasm/wasm-common';
import { TensorView } from '../../lib/wasm/jsep/tensor-view';
import { ComputeContext, ProgramInfo, TensorInfo } from '../../lib/wasm/jsep/webgpu/types';
import { parseSplitAttributes, split, SplitAttributes } from '../../lib/wasm/jsep/webgpu/ops/split';

function createTestGraphNode(name: string, opType: string): Graph.Node {
  return { name, opType, inputs: [], outputs: [], attributes: new Attribute(null) };
}

function dummyOpImpl(): Tensor[] {
  return [];
}

function checkConsistency(rules: readonly OpSet.ResolveRule[]) {
  const VERSION_MIN = 1,
    VERSION_MAX = 10;
  const typeRules = new Map<string, OpSet.ResolveRule[]>();
  rules.forEach((rule) => {
    let ruleSet = typeRules.get(rule[0]);
    if (!ruleSet) {
      ruleSet = [];
      typeRules.set(rule[0], ruleSet);
    }
    ruleSet.push(rule);
  });

  typeRules.forEach((rules, type) => {
    for (let i = VERSION_MIN; i < VERSION_MAX; i++) {
      let match = false;
      for (const r of rules) {
        try {
          resolveOperator(createTestGraphNode('', type), [{ domain: '', version: i }], [r]);
        } catch {
          continue;
        }
        if (match) {
          throw new Error(`multiple rules overlapped: opType='${type}', domain='', version=${i}`);
        }
        match = true;
      }
    }
  });
}

describe('#UnitTest# - resolveOperator', () => {
  const nodeAbs = createTestGraphNode('Abs_1', 'Abs');
  const opset7 = [{ domain: '', version: 7 }];
  it('ExpectFail - no rule available', () => {
    expect(() => {
      resolveOperator(nodeAbs, opset7, []);
    }).to.throw(TypeError);
  });
  it('ExpectFail - no matching rule', () => {
    expect(() => {
      resolveOperator(nodeAbs, opset7, [
        ['And', '', '7', dummyOpImpl],
        ['Sub', '', '7', dummyOpImpl],
      ]);
    }).to.throw(TypeError);
  });
  it('ExpectFail - version not match (exact match)', () => {
    expect(() => {
      resolveOperator(nodeAbs, opset7, [['Abs', '', '6', dummyOpImpl]]);
    }).to.throw(TypeError);
  });
  it('ExpectFail - version not match (minimum version match)', () => {
    expect(() => {
      resolveOperator(nodeAbs, opset7, [['Abs', '', '8+', dummyOpImpl]]);
    }).to.throw(TypeError);
  });
  it('ExpectFail - version not match (range match 1)', () => {
    expect(() => {
      resolveOperator(nodeAbs, opset7, [['Abs', '', '4-6', dummyOpImpl]]);
    }).to.throw(TypeError);
  });
  it('ExpectFail - version not match (range match 2)', () => {
    expect(() => {
      resolveOperator(nodeAbs, opset7, [['Abs', '', '8-10', dummyOpImpl]]);
    }).to.throw(TypeError);
  });
  it('ExpectPass - version match (exact match)', () => {
    resolveOperator(nodeAbs, opset7, [['Abs', '', '7', dummyOpImpl]]);
  });
  it('ExpectPass - version match (minimum version match)', () => {
    resolveOperator(nodeAbs, opset7, [['Abs', '', '5+', dummyOpImpl]]);
  });
  it('ExpectPass - version match (range match 1)', () => {
    resolveOperator(nodeAbs, opset7, [['Abs', '', '5-7', dummyOpImpl]]);
  });
  it('ExpectPass - version match (range match 2)', () => {
    resolveOperator(nodeAbs, opset7, [['Abs', '', '6-9', dummyOpImpl]]);
  });
});

describe('#UnitTest# - resolve rules', () => {
  const webglCheckOnlyRules = WEBGL_OP_RESOLVE_RULES.map(
    (rule) => [rule[0], rule[1], rule[2], dummyOpImpl] as OpSet.ResolveRule,
  );
  it('Consistency check - onnx.ai - webgl', () => {
    checkConsistency(webglCheckOnlyRules);
  });
});

describe('#UnitTest# - JSEP Split runtime shapes', () => {
  const runSplit = (
    dims: number[],
    attributes: SplitAttributes,
    splitSizes?: bigint[] | null,
  ): readonly TensorInfo[] => {
    const inputs = [{ dims, dataType: DataType.float }] as unknown as TensorView[];
    if (splitSizes !== undefined) {
      inputs.push({
        dims: splitSizes === null ? [] : [splitSizes.length],
        dataType: splitSizes === null ? 0 : DataType.int64,
        getBigInt64Array: () => BigInt64Array.from(splitSizes ?? []),
      } as unknown as TensorView);
    }
    let outputs: readonly TensorInfo[] = [];
    const context = {
      inputs,
      compute: (program: ProgramInfo) => {
        outputs = program.getRunData(inputs).outputs;
        return [];
      },
    } as unknown as ComputeContext;
    split(context, attributes);
    return outputs;
  };

  const attributes = parseSplitAttributes({ axis: 0, numOutputs: 2, splitSizes: [], isUnevenSplitAllowed: true });

  it('infers even split sizes without graph shapes', () => {
    expect(runSplit([4], attributes).map((output) => output.dims)).to.deep.equal([[2], [2]]);
  });

  it('ignores the empty optional split input placeholder', () => {
    expect(runSplit([4], attributes, null).map((output) => output.dims)).to.deep.equal([[2], [2]]);
  });

  it('recomputes uneven split sizes for changing input shapes', () => {
    expect(runSplit([5], attributes).map((output) => output.dims)).to.deep.equal([[3], [2]]);
    expect(runSplit([7], attributes).map((output) => output.dims)).to.deep.equal([[4], [3]]);
    expect(attributes.splitSizes).to.deep.equal([]);
  });

  it('normalizes a negative split axis', () => {
    const negativeAxis = parseSplitAttributes({ axis: -1, numOutputs: 2, splitSizes: [], isUnevenSplitAllowed: true });
    expect(runSplit([2, 5], negativeAxis).map((output) => output.dims)).to.deep.equal([
      [2, 3],
      [2, 2],
    ]);
  });

  it('preserves explicit split sizes from an input tensor', () => {
    expect(runSplit([5], attributes, [1n, 4n]).map((output) => output.dims)).to.deep.equal([[1], [4]]);
  });

  it('preserves explicit split sizes from an attribute', () => {
    const explicitSizes = parseSplitAttributes({ axis: 0, numOutputs: 2, splitSizes: [1, 4] });
    expect(runSplit([5], explicitSizes).map((output) => output.dims)).to.deep.equal([[1], [4]]);
  });

  it('requires even splitting without the num_outputs attribute', () => {
    const evenOnly = parseSplitAttributes({ axis: 0, numOutputs: 2, splitSizes: [] });
    expect(runSplit([4], evenOnly).map((output) => output.dims)).to.deep.equal([[2], [2]]);
    expect(() => runSplit([5], evenOnly)).to.throw('evenly divisible');
  });

  it('rejects invalid output counts', () => {
    const invalidCount = parseSplitAttributes({ axis: 0, numOutputs: 0, splitSizes: [], isUnevenSplitAllowed: true });
    expect(() => runSplit([4], invalidCount)).to.throw('numOutputs must be positive');
    expect(() => runSplit([1], attributes)).to.throw('must not exceed');
    const tooManyChunks = parseSplitAttributes({ axis: 0, numOutputs: 7, splitSizes: [], isUnevenSplitAllowed: true });
    expect(() => runSplit([10], tooManyChunks)).to.throw('nonempty splits');
  });
});
