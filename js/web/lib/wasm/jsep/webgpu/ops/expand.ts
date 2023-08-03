// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from '../types';

import {createIndicesHelper, ShaderHelper} from './common';

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
  const inputIndicesHelper = createIndicesHelper('input', inputShape);
  const outputIndicesHelper = createIndicesHelper('output', outputShape);
  const dataType = 'f32';

  const isl = inputShape.length;
  const osl = outputShape.length;
  const calculateInputIndexImpl = (): string => `
  fn calculateInputIndex(outputIndices: ${outputIndicesHelper.iType}) -> ${inputIndicesHelper.iType} {
    ${inputIndicesHelper.indicesVariableDeclaration('inputIndices')}
    for (var i = 0; i < ${isl}; i++) {
        if (inputShape[i] == 1) {
            // TODO: IndicesHelper should offer uniform way to get/set indices for all ranks
            inputIndices${isl >= 2 ? '[i]' : ''} = 0;
        } else {
            inputIndices${isl >= 2 ? '[i]' : ''} = ${osl > 1 ? `outputIndices[i + ${osl - isl}]` : 'outputIndices'};
        }
    }
    return inputIndices;
}`;

  const getShaderSource = (shaderHelper: ShaderHelper) => `
  const inputShape = array<u32, ${inputShape.length}>(${inputShape.map(i => `${i}u`).join(',')});
  ${calculateInputIndexImpl()};
  @group(0) @binding(0) var<storage, read> input : array<${dataType}>;
  @group(0) @binding(1) var<storage, read_write> output : array<${dataType}>;
  ${outputIndicesHelper.o2iImpl}
  ${inputIndicesHelper.i2oImpl}
  ${shaderHelper.mainStart()}
  ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
  ${inputIndicesHelper.indicesVariableDeclaration('inputIndices')}
  ${outputIndicesHelper.indicesVariableDeclaration('outputIndices')}
  ${outputIndicesHelper.o2iCall('global_idx', 'outputIndices')}
  inputIndices = calculateInputIndex(outputIndices);
  output[global_idx] = input[${inputIndicesHelper.i2oExpression('inputIndices')}];
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
  const cacheHint = context.inputs.map(x => x.dims.toString()).join('_');
  context.compute(
      {...expandProgramMetadata, cacheHint, get: () => createExpandProgramInfo(expandProgramMetadata, context.inputs)},
      {inputs: [0]});
};
