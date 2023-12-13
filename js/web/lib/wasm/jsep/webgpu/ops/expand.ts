// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {ComputeContext, ProgramInfo, ProgramUniform} from '../types';

import {createTensorShapeVariables, enableShapesUniforms, inputVariable, outputVariable, ShaderHelper} from './common';

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
  const dataType = inputs[0].dataType;
  const components = dataType === DataType.bool ? 4 : 1;
  const outputSize = ShapeUtil.size(outputShape) / components;

  const enableInputShapeUniform = enableShapesUniforms(inputShape.length);
  const enableOutputShapeUniform = enableShapesUniforms(outputShape.length);


  const getShaderSource = (shaderHelper: ShaderHelper) => {
    const inputShapeOrRank = enableInputShapeUniform ? inputShape.length : inputShape;
    const outputShapeOrRank = enableOutputShapeUniform ? outputShape.length : outputShape;
    const input = inputVariable('input', dataType, inputShapeOrRank, components);
    const output = outputVariable('output', dataType, outputShapeOrRank, components);
    let assignment: string;
    if (dataType === DataType.bool) {
      const singleAssignment = (resStr: string, x: number, typeCast = '') => `
          let outputIndices${x} = ${output.offsetToIndices(`outputOffset + ${x}u`)};
          let offset${x} = ${input.broadcastedIndicesToOffset(`outputIndices${x}`, output)};
          let index${x} = offset${x} / 4u;
          let component${x} = offset${x} % 4u;
          ${resStr}[${x}] = ${typeCast}(${input.getByOffset(`index${x}`)}[component${x}]);
        `;
      assignment = `
        let outputOffset = global_idx * ${components};
        var data = vec4<u32>(0);
        ${singleAssignment('data', 0, 'u32')}
        ${singleAssignment('data', 1, 'u32')}
        ${singleAssignment('data', 2, 'u32')}
        ${singleAssignment('data', 3, 'u32')}
        ${output.setByOffset('global_idx', 'data')}
      }`;
    } else {
      assignment = `
        let outputIndices = ${output.offsetToIndices('global_idx')};
        let inputOffset = ${input.broadcastedIndicesToOffset('outputIndices', output)};
        ${output.setByOffset('global_idx', input.getByOffset('inputOffset'))}
      }`;
    }
    return `
    ${shaderHelper.registerUniform('vec_size', 'u32').declareVariables(input, output)}
    ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.vec_size')}
    ${assignment}`;
  };

  const programUniforms: ProgramUniform[] = [{type: 'uint32', data: outputSize}];
  if (enableInputShapeUniform) {
    programUniforms.push(...createTensorShapeVariables(inputShape));
  }
  if (enableOutputShapeUniform) {
    programUniforms.push(...createTensorShapeVariables(outputShape));
  }
  return {
    name: 'Expand',
    shaderCache: {hint: `${outputShape.length}`, inputDependencies: [enableInputShapeUniform ? 'rank' : 'dims']},
    getShaderSource,
    getRunData: () => ({
      outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
      dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)},
      programUniforms
    })
  };
};

export const expand = (context: ComputeContext): void => {
  validateInputs(context.inputs);
  context.compute(createExpandProgramInfo(context.inputs), {inputs: [0]});
};
