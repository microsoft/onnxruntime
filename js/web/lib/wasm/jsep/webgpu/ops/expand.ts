// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from '../types';

import {inputVariable, outputVariable, ShaderHelper} from './common';

export const expandProgramMetadata = {
  name: 'Expand',
  inputTypes: [GpuDataType.default]
};

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length !== 2) {
    throw new Error('Expand requires 2 input.');
  }
  const inputShape = inputs[0].dims;

  const shape: number[] = [];
  if (inputs[1].dims[0] > 0) {
    inputs[1].getBigInt64Array().forEach(v => shape.push(Number(v)));
  }
  let shapeIndex = shape.length < inputShape.length ? 0 : shape.length - inputShape.length;
  let inputShapeIndex = inputShape.length < shape.length ? 0 : inputShape.length - shape.length;
  for (; shapeIndex < shape.length && inputShapeIndex < inputShape.length; ++shapeIndex, ++inputShapeIndex) {
    if (shape[shapeIndex] !== inputShape[inputShapeIndex] && shape[shapeIndex] !== 1 &&
        inputShape[inputShapeIndex] !== 1) {
      throw new Error('Expand requires shape to be broadcastable to input');
    }
  }
};

const getAdjustedShape = (shape1: readonly number[], shape2: readonly number[]): number[] => {
  const diff = shape1.length - shape2.length;
  const shape: number[] = [];
  for (let i = 0; i < diff; ++i) {
    shape.push(shape1[i]);
  }
  for (let i = 0; i < shape2.length; ++i) {
    shape.push(shape2[i] === 1 ? shape1[i + diff] : shape2[i]);
  }
  return shape;
};

const calculateOutputShape = (inputShape: readonly number[], shape: readonly number[]): number[] =>
    (inputShape.length > shape.length) ? getAdjustedShape(inputShape, shape) : getAdjustedShape(shape, inputShape);


const createExpandProgramInfo = (metadata: ProgramMetadata, inputs: readonly TensorView[]): ProgramInfo => {
  const inputShape = inputs[0].dims;

  const shape: number[] = [];
  if (inputs[1].dims[0] > 0) {
    inputs[1].getBigInt64Array().forEach(v => shape.push(Number(v)));
  }
  const outputShape: number[] = calculateOutputShape(inputShape, shape);
  const outputSize = ShapeUtil.size(outputShape);

  const dataType = 'f32';
  const input = inputVariable('input', dataType, inputShape);
  const output = outputVariable('output', dataType, outputShape);

  const calculateInputIndexImpl = (): string => `
  fn calculateInputIndex(outputIndices: array<u32, ${outputShape.length}>) -> array<u32,${inputShape.length}> {
    ${input.indicesVariableDeclaration('inputIndices')}
    for (var i = 0; i < ${inputShape.length}; i++) {
        if (inputShape[i] == 1) {
            inputIndices[i] = 0;
        } else {
            inputIndices[i] = outputIndices[i + ${outputShape.length - inputShape.length}];
        }
    }
    return inputIndices;
}`;

  const getShaderSource = (shaderHelper: ShaderHelper) => `
  const inputShape = array<u32, ${inputShape.length}>(${inputShape.map(i => `${i}u`).join(',')});
  ${calculateInputIndexImpl()};
  @group(0) @binding(0) var<storage, read> input : array<${dataType}>;
  @group(0) @binding(1) var<storage, read_write> output : array<${dataType}>;
  ${output.offsetToIndicesImplementation}
  ${input.indicesToOffsetImplementation}
  ${shaderHelper.mainStart()}
  ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
  ${input.indicesVariableDeclaration('inputIndices')}
  ${output.indicesVariableDeclaration('outputIndices')}
  ${output.offsetToIndices('global_idx', 'outputIndices')}
  inputIndices = calculateInputIndex(outputIndices);
  output[global_idx] = input[${input.indicesToOffset('inputIndices')}];
}`;
  return {
    ...metadata,
    getShaderSource,
    outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
    dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
  };
};

export const expand = (context: ComputeContext): void => {
  validateInputs(context.inputs);
  context.compute(
      {...expandProgramMetadata, get: () => createExpandProgramInfo(expandProgramMetadata, context.inputs)},
      {inputs: [0]});
};
