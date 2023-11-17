// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo, TensorInfo} from '../types';

import {IndicesHelper, inputVariable, outputVariable, ShaderHelper} from './common';

export interface SplitAttributes extends AttributeWithCacheKey {
  readonly axis: number;
  readonly numOutputs: number;
  readonly splitSizes: number[];
}

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length < 1) {
    throw new Error('too few inputs');
  }
};

const createSplitAttributesFromInputs =
    (inputs: readonly TensorView[], attributes: SplitAttributes): SplitAttributes => {
      const splitSizes: number[] = [];
      let numOutputs: number = attributes.numOutputs;
      if (inputs[1].dims[0] > 0) {
        inputs[1].getBigInt64Array().forEach(v => splitSizes.push(Number(v)));
        numOutputs = splitSizes.length;
      }
      return createAttributeWithCacheKey({numOutputs, axis: attributes.axis, splitSizes});
    };

const calculateOutputIndexImpl = (numberOfTensors: number): string => `
fn calculateOutputIndex(index: u32) -> u32 {
    for (var i: u32 = 0u; i < ${numberOfTensors}u; i += 1u ) {
    if (index < sizeInConcatAxis[i]) {
        return i;
    }
    }
    return ${numberOfTensors}u;
}`;
const writeBufferDataImpl = (outputs: readonly IndicesHelper[]) => {
  const numberOfTensors = outputs.length;
  const codeLines: string[] = [];
  for (let i = 0; i < numberOfTensors; ++i) {
    const returnSnippet = outputs[i].setByIndices('indices', 'input[global_idx]');
    if (numberOfTensors === 1) {
      codeLines.push(returnSnippet);
    } else if (i === 0) {
      codeLines.push(`if (outputNumber == ${i}u) { ${returnSnippet} }`);
    } else if (i === numberOfTensors - 1) {
      codeLines.push(`else { ${returnSnippet} }`);
    } else {
      codeLines.push(`else if (outputNumber == ${i}) { ${returnSnippet} }`);
    }
  }
  return `
      fn writeBufferData(outputNumber: u32, indices: ${outputs[0].type.indices}, global_idx: u32) {
        ${codeLines.join('\n')}
      }`;
};

const createSplitProgramInfo = (inputs: readonly TensorView[], attributes: SplitAttributes): ProgramInfo => {
  const inputShape = inputs[0].dims;
  const inputSize = ShapeUtil.size(inputShape);
  const dataType = inputs[0].dataType;
  const rank = inputShape.length;
  const axis = attributes.axis;
  const adjustedAxis = (axis < 0) ? inputShape.length + axis : axis;
  const outputs = new Array<IndicesHelper>(attributes.numOutputs);
  const input = inputVariable('input', dataType, inputShape);
  const sizeInConcatAxis = new Array<number>(attributes.numOutputs);
  const outputsTensorInfo: TensorInfo[] = [];
  const outputShapes: number[][] = [];
  let previousSum = 0;
  for (let i = 0; i < attributes.numOutputs; i++) {
    previousSum += attributes.splitSizes[i];
    sizeInConcatAxis[i] = previousSum;
    const outputShape = inputShape.slice();
    outputShape[attributes.axis] = attributes.splitSizes[i];
    outputShapes.push(outputShape);
    outputs[i] = outputVariable(`output${i}`, dataType, outputShapes[i]);
    outputsTensorInfo.push({dims: outputShapes[i], dataType: inputs[0].dataType});
  }
  const indicesAxis = rank < 2 ? 'indices' : `indices[${adjustedAxis}]`;
  const getShaderSource = (shaderHelper: ShaderHelper) => `
  ${shaderHelper.declareVariables(input, ...outputs)}
  const sizeInConcatAxis = array<u32, ${sizeInConcatAxis.length}>(${sizeInConcatAxis.map(i => `${i}u`).join(',')});
  ${calculateOutputIndexImpl(sizeInConcatAxis.length)}
  ${writeBufferDataImpl(outputs)}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(inputSize)}

    var indices = ${input.offsetToIndices('global_idx')};
    let outputNumber = calculateOutputIndex(${indicesAxis});
    if (outputNumber != 0) {
        ${indicesAxis} -= sizeInConcatAxis[outputNumber - 1u];
    }
    writeBufferData(outputNumber, indices, global_idx);
  }`;
  return {
    name: 'Split',
    shaderCache: {hint: attributes.cacheKey},
    getShaderSource,
    getRunData: () => ({
      outputs: outputsTensorInfo,
      dispatchGroup: {x: Math.ceil(inputSize / 64 /* workgroup size */)},
    })
  };
};

export const split = (context: ComputeContext, attributes: SplitAttributes): void => {
  validateInputs(context.inputs);
  const updatedAttributes =
      context.inputs.length === 1 ? attributes : createSplitAttributesFromInputs(context.inputs, attributes);
  context.compute(createSplitProgramInfo(context.inputs, updatedAttributes), {inputs: [0]});
};

export const parseSplitAttributes = (attributes: Record<string, unknown>): SplitAttributes => {
  const axis = attributes.axis as number;
  const splitSizes: number[] = attributes.splitSizes as number[];
  const numOutputs = attributes.numOutputs as number < 0 ? splitSizes.length : attributes.numOutputs as number;
  if (numOutputs !== splitSizes.length) {
    throw new Error('numOutputs and splitSizes lengh must be equal');
  }
  return createAttributeWithCacheKey({axis, numOutputs, splitSizes});
};
