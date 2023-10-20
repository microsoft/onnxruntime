// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo} from '../types';

import {IndicesHelper, inputVariable, outputVariable, ShaderHelper} from './common';

export interface ConcatAttributes extends AttributeWithCacheKey {
  readonly axis: number;
}

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length < 1) {
    throw new Error('too few inputs');
  }

  const inputType = inputs[0].dataType;
  const inputDimensionality = inputs[0].dims.length;

  for (const input of inputs) {
    // make sure types of all inputs match
    if (input.dataType !== inputType) {
      throw new Error('input tensors should be one type');
    }

    // make sure the dimensionality of all inputs are the same
    if (input.dims.length !== inputDimensionality) {
      throw new Error('input tensors should have the same shape');
    }
  }
};

const calculateInputIndexImpl = (numberOfTensors: number): string => `
  fn calculateInputIndex(index: u32) -> u32 {
    for (var i: u32 = 0u; i < ${numberOfTensors}u; i += 1u ) {
      if (index < sizeInConcatAxis[i]) {
        return i;
      }
    }
    return ${numberOfTensors}u;
  }`;

const assignOutputData = (inputs: readonly IndicesHelper[], output: IndicesHelper) => {
  const numberOfTensors = inputs.length;

  const codeLines: string[] = [];
  for (let i = 0; i < numberOfTensors; ++i) {
    const returnSnippet = output.setByOffset('global_idx', inputs[i].getByIndices('indices'));
    if (numberOfTensors === 1) {
      codeLines.push(returnSnippet);
    } else if (i === 0) {
      codeLines.push(`if (inputIndex == ${i}u) { ${returnSnippet} }`);
    } else if (i === numberOfTensors - 1) {
      codeLines.push(`else { ${returnSnippet} }`);
    } else {
      codeLines.push(`else if (inputIndex == ${i}) { ${returnSnippet} }`);
    }
  }
  return codeLines.join('\n');
};

const createConcatProgramInfo = (inputs: readonly TensorView[], axis: number): ProgramInfo => {
  const inputShape = inputs[0].dims.slice();
  if (axis >= inputShape.length || axis < (-1 * inputShape.length)) {
    throw new Error('axis specified for concat doesn\'t match input dimensionality');
  }
  const adjustedAxis = (axis < 0) ? inputShape.length + axis : axis;
  // ensure all of the non-concatenated axes match each other
  // calculate the shape of the output tensor while we do that
  const outputShape = inputShape.slice(0);
  for (let i = 1; i < inputs.length; i++) {
    const dataNShape = inputs[i].dims.slice();
    for (let axisIndex = 0; axisIndex < inputShape.length; axisIndex++) {
      // add to the placeholder for computing output shape
      if (axisIndex === adjustedAxis) {
        outputShape[adjustedAxis] += dataNShape[axisIndex];
      }
      // ensure all non-cancatenated axes match each other
      else if (inputShape[axisIndex] !== dataNShape[axisIndex]) {
        throw new Error('non concat dimensions must match');
      }
    }
  }

  const outputSize = ShapeUtil.size(outputShape);

  const sizeInConcatAxis = new Array<number>(inputs.length);
  const inputVars = new Array<IndicesHelper>(inputs.length);
  const dataType = inputs[0].dataType;

  let previousSum = 0;
  for (let i = 0; i < inputs.length; ++i) {
    previousSum += inputs[i].dims[adjustedAxis];
    sizeInConcatAxis[i] = previousSum;

    inputVars[i] = inputVariable(`input${i}`, dataType, inputs[i].dims);
  }

  const output = outputVariable('output', dataType, outputShape);

  const indicesAxis = output.indicesGet('indices', adjustedAxis);
  const getShaderSource = (shaderHelper: ShaderHelper) => `
  ${shaderHelper.declareVariables(...inputVars, output)}

  const sizeInConcatAxis = array<u32, ${sizeInConcatAxis.length}>(${sizeInConcatAxis.map(i => `${i}u`).join(',')});
  ${calculateInputIndexImpl(sizeInConcatAxis.length)}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

    var indices = ${output.offsetToIndices('global_idx')};

    let inputIndex = calculateInputIndex(${indicesAxis});
    if (inputIndex != 0u) {
      ${indicesAxis} -= sizeInConcatAxis[inputIndex - 1u];
    }

    ${assignOutputData(inputVars, output)}
  }`;
  return {
    name: 'Concat',
    shaderCache: {hint: `${axis}`},
    getRunData: () => ({
      outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
      dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)}
    }),
    getShaderSource,
  };
};

export const concat = (context: ComputeContext, attributes: ConcatAttributes): void => {
  validateInputs(context.inputs);
  context.compute(createConcatProgramInfo(context.inputs, attributes.axis));
};

export const parseConcatAttributes = (attributes: Record<string, unknown>): ConcatAttributes =>
    createAttributeWithCacheKey({axis: attributes.axis as number});
