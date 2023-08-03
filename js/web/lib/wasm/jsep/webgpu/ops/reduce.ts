// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {inputVariable, outputVariable, ShaderHelper} from './common';

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length === 0 || inputs.length > 2) {
    throw new Error('Reduce op requires 1 or 2 inputs.');
  }

  if (inputs.length === 2 && inputs[1].dims.length !== 1) {
    throw new Error('Invalid axes input dims.');
  }

  if (inputs[0].dataType !== DataType.float) {
    throw new Error('Invalid input type.');
  }
};

export interface ReduceAttributes extends AttributeWithCacheKey {
  keepDims: boolean;
  noopWithEmptyAxes: boolean;
  axes: number[];
}

type ReduceOp = (inputs: readonly TensorView[], axes: number[]) => string[];
const noOp: ReduceOp = (): string[] => ['', '', 'value = _A[inputIdx];', ''];
const createReduceProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: ReduceAttributes, reduceOp: ReduceOp):
        ProgramInfo => {
          const outputShape: number[] = [];
          const inputShape = inputs[0].dims;

          const idxCopy: string[] = [];  // copy output indexes to input indexes

          const axes = ShapeUtil.normalizeAxes(attributes.axes, inputs[0].dims.length);
          const outputDimsLength = inputs[0].dims.length - (attributes.keepDims ? 0 : axes.length);
          const ops = reduceOp(inputs, axes);
          const input = inputVariable('_A', inputs[0].dataType, inputShape);
          const output = outputVariable('output', inputs[0].dataType, outputShape);
          const initInputIdxLet = `let inputIdx = ${input.indicesToOffset('inputIndices')};`;
          const initInputIdxVar = `var inputIdx = ${input.indicesToOffset('inputIndices')};`;
          const updateInputIdxImpl = `inputIdx = ${input.indicesToOffset('inputIndices')};`;
          const initInputIdx = (ops[1] === '') ? '' : initInputIdxVar;
          let reduceOps = ((ops[1] === '') ? initInputIdxLet : updateInputIdxImpl) + '\n' + ops[2];
          const reduceOnAllAxes = !attributes.noopWithEmptyAxes && attributes.axes.length === 0;
          for (let k = 0; k < inputs[0].dims.length; k++) {
            // if this axis is reduced
            if (reduceOnAllAxes || axes.indexOf(k) >= 0) {
              if (attributes.keepDims) {
                outputShape.push(1);
              }  // else { remove the axis from outputShape; }

              // loop over the d-th axis
              reduceOps = `for(var j${k}: u32 = 0; j${k} < ${inputs[0].dims[k]}; j${k}++) {
                  ${input.indicesSet('inputIndices', k, `j${k}`)}
                  ${reduceOps}
                }`;
            } else {
              const outputIndices = outputDimsLength > 1 ? `outputIndices[${outputShape.length}]` : 'outputIndices';
              idxCopy.push(
                  `${input.indicesSet('inputIndices', k, output.indicesGet('outputIndices', outputShape.length))} = ${
                      outputIndices};`);
              outputShape.push(inputs[0].dims[k]);
            }
          }

          const outputSize = ShapeUtil.size(outputShape);

          const getShaderSource = (shaderHelper: ShaderHelper) => `
          ${shaderHelper.declareVariables(input, output)}

          ${output.impl('offsetToIndices')}
          ${input.impl('indicesToOffset')}

          ${shaderHelper.mainStart()}
          ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
          let outputIndices = ${output.offsetToIndices('global_idx')};
          var inputIndices: ${input.type.indices};

          var value = ${output.type.value}(0);

          ${idxCopy.join('\n')}
          ${ops[0]}       // init ops for reduce max/min
          ${initInputIdx}
          ${ops[1]}
          ${reduceOps}
          ${ops[3]}       // final computation for reduce mean
          output[global_idx] = value;
        }`;

          return {
            ...metadata,
            getShaderSource,
            outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
            dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
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

const createReduceProgramInfoLoader =
    (inputs: readonly TensorView[], name: string, attributes: ReduceAttributes, reduceOp: ReduceOp):
        ProgramInfoLoader => {
          const updatedAttributes: ReduceAttributes =
              inputs.length === 1 ? attributes : createReduceAttributesFromInputs(inputs, attributes);
          const metadata: ProgramMetadata = {
            name,
            inputTypes: [GpuDataType.default],
            cacheHint: updatedAttributes.cacheKey + '_' + inputs[0].dims.map(d => d.toString()).join(',')
          };
          return {
            ...metadata,
            get: () => createReduceProgramInfo(
                metadata, [inputs[0]], updatedAttributes,
                updatedAttributes.noopWithEmptyAxes && updatedAttributes.axes.length === 0 ? noOp : reduceOp)
          };
        };

export const reduceLogSum = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (): string[] => ['value = 0.0;', '', 'value += _A[inputIdx];', 'value = log(value);'];
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceLogSum', attributes, reduceOp), {inputs: [0]});
};

export const reduceL1 = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (): string[] => ['value = 0.0;', '', 'value += abs(_A[inputIdx]);', ''];
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceL1', attributes, reduceOp), {inputs: [0]});
};

export const reduceL2 = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = ():
      string[] => ['var t = f32(0); value = 0.0;', '', 't = _A[inputIdx]; value += (t * t);', 'value = sqrt(value);'];
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceL2', attributes, reduceOp), {inputs: [0]});
};

export const reduceLogSumExp = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (): string[] => ['value = 0.0;', '', 'value += exp(_A[inputIdx]);', 'value = log(value);'];
  context.compute(
      createReduceProgramInfoLoader(context.inputs, 'ReduceLogSumExp', attributes, reduceOp), {inputs: [0]});
};

export const reduceMax = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (inputs: TensorView[], axes: number[]): string[] => {
    const idxZero = [];
    for (let k = 0; k < inputs[0].dims.length; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        idxZero.push(`inputIndices[${k}] = 0;`);  // first element
      }
    }

    return [`${idxZero.join('\n')}`, 'value = _A[inputIdx];', 'value = max(value, _A[inputIdx]);', ''];
  };
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceMax', attributes, reduceOp), {inputs: [0]});
};

export const reduceMean = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (inputs: TensorView[], axes: number[]): string[] => {
    let size = 1.0;
    for (let k = 0; k < inputs[0].dims.length; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        size *= inputs[0].dims[k];
      }
    }

    return ['value = 0.0;', '', 'value += _A[inputIdx];', `value = value / ${size}.;`];  // ensure real number with `.`
  };
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceMean', attributes, reduceOp), {inputs: [0]});
};

export const reduceMin = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (inputs: TensorView[], axes: number[]): string[] => {
    const idxZero = [];
    for (let k = 0; k < inputs[0].dims.length; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        idxZero.push(`inputIndices[${k}] = 0;`);  // first element
      }
    }

    return [`${idxZero.join('\n')}`, 'value = _A[inputIdx];', 'value = min(value, _A[inputIdx]);', ''];
  };
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceMin', attributes, reduceOp), {inputs: [0]});
};

export const reduceProd = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (): string[] => ['value = 1.0;', '', 'value *= _A[inputIdx];', ''];
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceProd', attributes, reduceOp), {inputs: [0]});
};

export const reduceSum = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (): string[] => ['value = 0.0;', '', 'value += _A[inputIdx];', ''];
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceSum', attributes, reduceOp), {inputs: [0]});
};

export const reduceSumSquare = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp =
      (): string[] => ['var t = f32(0); value = 0.0;', '', 't = _A[inputIdx]; value += t * t;', ''];
  context.compute(
      createReduceProgramInfoLoader(context.inputs, 'ReduceSumSquare', attributes, reduceOp), {inputs: [0]});
};

export const parseReduceAttributes = (attributes: Record<string, unknown>): ReduceAttributes =>
    createAttributeWithCacheKey(attributes as Omit<ReduceAttributes, keyof AttributeWithCacheKey>);
