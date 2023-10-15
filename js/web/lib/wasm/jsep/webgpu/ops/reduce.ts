// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo, ProgramShaderCacheInfo} from '../types';

import {IndicesHelper, inputVariable, outputVariable, ShaderHelper} from './common';

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length === 0 || inputs.length > 2) {
    throw new Error('Reduce op requires 1 or 2 inputs.');
  }

  if (inputs.length === 2 && inputs[1].dims.length !== 1) {
    throw new Error('Invalid axes input dims.');
  }
};

export interface ReduceAttributes extends AttributeWithCacheKey {
  keepDims: boolean;
  noopWithEmptyAxes: boolean;
  axes: number[];
}

export type ReduceOp =
    (input: IndicesHelper, output: IndicesHelper,
     axes: readonly number[]) => [string, string, string, string, ...string[]];

const noOp: ReduceOp = (input) => ['', '', `var value = ${input.getByOffset('inputOffset')};`, ''];
export const createReduceProgramInfo =
    (name: string, shaderCache: ProgramShaderCacheInfo, inputs: readonly TensorView[], reduceOp: ReduceOp,
     axesInput: number[], outputDataType: DataType, keepDims = false, noopWithEmptyAxes = false): ProgramInfo => {
      const outputShape: number[] = [];
      const inputShape = inputs[0].dims;

      const axes = ShapeUtil.normalizeAxes(axesInput, inputs[0].dims.length);
      const reduceOnAllAxes = !noopWithEmptyAxes && axes.length === 0;
      inputShape.forEach((d, i) => {
        if (reduceOnAllAxes || axes.indexOf(i) >= 0) {
          if (keepDims) {
            outputShape.push(1);
          }  // else { // skip this axis}
        } else {
          outputShape.push(d);
        }
      });

      const idxCopy: string[] = [];  // copy output indexes to input indexes

      const input = inputVariable('_A', inputs[0].dataType, inputShape);
      const output = outputVariable('output', outputDataType, outputShape);
      const ops = reduceOp(input, output, axes);
      const inputOffsetAssignment = `inputOffset = ${input.indicesToOffset('inputIndices')};`;
      const initinputOffsetLet = `let ${inputOffsetAssignment};`;
      const initinputOffsetVar = `var ${inputOffsetAssignment};`;
      const initinputOffset = (ops[1] === '') ? '' : initinputOffsetVar;
      let reduceOps = ((ops[1] === '') ? initinputOffsetLet : inputOffsetAssignment) + '\n' + ops[2];

      for (let k = 0, l = 0; k < inputs[0].dims.length; k++) {
        // if this axis is reduced
        if (reduceOnAllAxes || axes.indexOf(k) >= 0) {
          if (keepDims) {
            l++;
          }
          // loop over the d-th axis
          reduceOps = `for(var j${k}: u32 = 0; j${k} < ${inputs[0].dims[k]}; j${k}++) {
                ${ops[2].includes('lastIndex') ? `let lastIndex = j${k};` : ''}
                ${input.indicesSet('inputIndices', k, `j${k}`)}
                ${reduceOps}
              }`;
        } else {
          idxCopy.push(`${input.indicesSet('inputIndices', k, output.indicesGet('outputIndices', l))};`);
          l++;
        }
      }

      const outputSize = ShapeUtil.size(outputShape);
      const getShaderSource = (shaderHelper: ShaderHelper) => `
        ${shaderHelper.declareVariables(input, output)}

        ${shaderHelper.mainStart()}
          ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
          var inputIndices: ${input.type.indices};
          let outputIndices = ${output.offsetToIndices('global_idx')};

          ${idxCopy.join('\n')}
          ${ops[0]}       // init ops for reduce max/min
          ${initinputOffset}
          ${ops[1]}
          ${reduceOps}
          ${ops[3]}
          ${ops.length === 4 ? output.setByOffset('global_idx', 'value') : ops.slice(4).join('\n')}
        }`;

      return {
        name,
        shaderCache,
        getShaderSource,
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: outputDataType}],
          dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)}
        }),
      };
    };

const createReduceAttributesFromInputs =
    (inputs: readonly TensorView[], attributes: ReduceAttributes): ReduceAttributes => {
      const axes: number[] = [];
      if (inputs[1].dims[0] > 0) {
        inputs[1].getBigInt64Array().forEach(v => axes.push(Number(v)));
      }
      return createAttributeWithCacheKey(
          {axes, keepDims: attributes.keepDims, noopWithEmptyAxes: attributes.noopWithEmptyAxes});
    };

const runReduceProgram =
    (context: ComputeContext, name: string, attributes: ReduceAttributes, reduceOp: ReduceOp): void => {
      const inputs = context.inputs;
      const updatedAttributes: ReduceAttributes =
          inputs.length === 1 ? attributes : createReduceAttributesFromInputs(inputs, attributes);

      context.compute(
          createReduceProgramInfo(
              name, {hint: updatedAttributes.cacheKey}, [inputs[0]],
              updatedAttributes.noopWithEmptyAxes && updatedAttributes.axes.length === 0 ? noOp : reduceOp,
              updatedAttributes.axes, inputs[0].dataType, updatedAttributes.keepDims,
              updatedAttributes.noopWithEmptyAxes),
          {inputs: [0]});
    };

export const reduceLogSum = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, output) =>
      [`var value = ${output.type.storage}(0);`,
       '',
       `value += ${input.getByOffset('inputOffset')};`,
       'value = log(value);',
  ];
  runReduceProgram(context, 'ReduceLogSum', attributes, reduceOp);
};

export const reduceL1 = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, output) =>
      [`var value = ${output.type.storage}(0);`,
       '',
       `value += abs(${input.getByOffset('inputOffset')});`,
       '',
  ];
  runReduceProgram(context, 'ReduceL1', attributes, reduceOp);
};

export const reduceL2 = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, output) =>
      [`var t = ${output.type.value}(0); var value = ${output.type.value}(0);`,
       '',
       `t = ${input.getByOffset('inputOffset')}; value += (t * t);`,
       'value = sqrt(value);',
  ];
  runReduceProgram(context, 'ReduceL2', attributes, reduceOp);
};

export const reduceLogSumExp = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, output) =>
      [`var value = ${output.type.storage}(0);`,
       '',
       `value += exp(${input.getByOffset('inputOffset')});`,
       'value = log(value);',
  ];
  runReduceProgram(context, 'ReduceLogSumExp', attributes, reduceOp);
};

export const reduceMax = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, _output, axes) => {
    const idxZero = [];
    for (let k = 0; k < input.rank; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        idxZero.push(input.indicesSet('inputIndices', k, 0));
      }
    }

    return [
      `${idxZero.join('\n')}`,
      `var value = ${input.getByOffset('inputOffset')};`,
      `value = max(value, ${input.getByOffset('inputOffset')});`,
      '',
    ];
  };
  runReduceProgram(context, 'ReduceMax', attributes, reduceOp);
};

export const reduceMean = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, output, axes) => {
    let size = 1.0;
    for (let k = 0; k < input.rank; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        // TODO: this depends on the input dims. If we want to use uniform, this need to be updated.
        size *= context.inputs[0].dims[k];
      }
    }

    return [
      'var sum = f32(0);',
      '',
      `sum += f32(${input.getByOffset('inputOffset')});`,
      `let value = ${output.type.value}(sum / ${size});`,
    ];
  };
  runReduceProgram(context, 'ReduceMean', attributes, reduceOp);
};

export const reduceMin = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, _output, axes) => {
    const idxZero = [];
    for (let k = 0; k < input.rank; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        idxZero.push(`inputIndices[${k}] = 0;`);  // first element
      }
    }

    return [
      `${idxZero.join('\n')}`,
      `var value = ${input.getByOffset('inputOffset')};`,
      `value = min(value, ${input.getByOffset('inputOffset')});`,
      '',
    ];
  };
  runReduceProgram(context, 'ReduceMin', attributes, reduceOp);
};

export const reduceProd = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, output) =>
      [`var value = ${output.type.storage}(1);`,
       '',
       `value *= ${input.getByOffset('inputOffset')};`,
       '',
  ];
  runReduceProgram(context, 'ReduceProd', attributes, reduceOp);
};

export const reduceSum = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, output) =>
      [`var value = ${output.type.storage}(0);`,
       '',
       `value += ${input.getByOffset('inputOffset')};`,
       '',
  ];
  runReduceProgram(context, 'ReduceSum', attributes, reduceOp);
};

export const reduceSumSquare = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, output) =>
      [`var t = ${output.type.value}(0); var value = ${output.type.value}(0);`,
       '',
       `t = ${input.getByOffset('inputOffset')}; value += t * t;`,
       '',
  ];
  runReduceProgram(context, 'ReduceSumSquare', attributes, reduceOp);
};

export const parseReduceAttributes = (attributes: Record<string, unknown>): ReduceAttributes =>
    createAttributeWithCacheKey(attributes as Omit<ReduceAttributes, keyof AttributeWithCacheKey>);
