// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {ComputeContext, ProgramInfo} from '../types';

import {inputVariable, outputVariable, ShaderHelper} from './common';

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length !== 2) {
    throw new Error('Expand requires 2 input.');
  }
  const inputShape = inputs[0].dims;
  const shape = Array.from(inputs[1].getBigInt64Array(), Number);

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


const createExpandProgramInfo = (inputs: readonly TensorView[]): ProgramInfo => {
  const inputShape = inputs[0].dims;
  const shape = Array.from(inputs[1].getBigInt64Array(), Number);
  const outputShape: number[] = calculateOutputShape(inputShape, shape);
  const outputSize = ShapeUtil.size(outputShape);

  const dataType = inputs[0].dataType;
  const input = inputVariable('input', dataType, inputShape);
  const output = outputVariable('output', dataType, outputShape);

  const getShaderSource = (shaderHelper: ShaderHelper) => `
  const inputShape = ${input.indices(...inputShape)};
  ${shaderHelper.declareVariables(input, output)}
  ${shaderHelper.mainStart()}
  ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
    let outputIndices = ${output.offsetToIndices('global_idx')};
    var inputIndices: ${input.type.indices};
    for (var i = 0; i < ${inputShape.length}; i++) {
      if (${input.indicesGet('inputShape', 'i')} == 1) {
        ${input.indicesSet('inputIndices', 'i', 0)}
      } else {
        ${
      input.indicesSet(
          'inputIndices', 'i', output.indicesGet('outputIndices', `i + ${outputShape.length - inputShape.length}`))}
      }
    }
    ${output.setByOffset('global_idx', input.getByIndices('inputIndices'))}
  }`;
  return {
    name: 'Expand',
    shaderCache: {hint: `${outputShape}`},
    getShaderSource,
    getRunData: () => ({
      outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
      dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)}
    })
  };
};

export const expand = (context: ComputeContext): void => {
  validateInputs(context.inputs);
  context.compute(createExpandProgramInfo(context.inputs), {inputs: [0]});
};
