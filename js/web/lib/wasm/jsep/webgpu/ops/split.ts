// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata, TensorInfo} from '../types';

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
      if (inputs[1].dims[0] > 0) {
        inputs[1].getBigInt64Array().forEach(v => splitSizes.push(Number(v)));
      }
      return createAttributeWithCacheKey({numOutputs: attributes.numOutputs, axis: attributes.axis, splitSizes});
    };

const calculateOutputIndexImpl = (numberOfTensors: number): string => `
fn calculateOutputIndex(index: i32) -> i32 {
    for (var i: i32 = 0; i < ${numberOfTensors}; i++) {
      if (index < sizeInConcatAxis[i]) {
          return i;
      }
    }
    return ${numberOfTensors};
}`;
const writeOutputData = (outputs: readonly IndicesHelper[]) => {
  const numberOfTensors = outputs.length;
  const codeLines: string[] = [];
  for (let i = 0; i < numberOfTensors; ++i) {
    const returnSnippet = outputs[i].setByIndices('indices', 'input[global_idx]');
    if (numberOfTensors === 1) {
      codeLines.push(returnSnippet);
    } else if (i === 0) {
      codeLines.push(`if (outputNumber == ${i}) { ${returnSnippet} }`);
    } else if (i === numberOfTensors - 1) {
      codeLines.push(`else { ${returnSnippet} }`);
    } else {
      codeLines.push(`else if (outputNumber == ${i}) { ${returnSnippet} }`);
    }
  }
  return codeLines.join('\n');
};

const createSplitProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: SplitAttributes): ProgramInfo => {
      const inputShape = inputs[0].dims;
      const inputSize = ShapeUtil.size(inputShape);
      const dataType = inputs[0].dataType;
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
        outputsTensorInfo.push({dims: outputShapes[i], dataType: inputs[0].dataType, gpuDataType: GpuDataType.default});
      }
      const indicesAxis = input.indicesGet('indices', adjustedAxis);
      const getShaderSource = (shaderHelper: ShaderHelper) => `
  ${shaderHelper.declareVariables(input, ...outputs)}
  ${input.impl('indicesToOffset', 'offsetToIndices', 'get')}
  ${outputs.map(o => o.impl('indicesToOffset', 'set')).join('\n')}
  const sizeInConcatAxis = array<i32, ${sizeInConcatAxis.length}>(${sizeInConcatAxis.join(',')});
  ${calculateOutputIndexImpl(sizeInConcatAxis.length)}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(inputSize)}

    var indices = ${input.offsetToIndices('global_idx')};
    let outputNumber = calculateOutputIndex(${indicesAxis});
    if (outputNumber != 0) {
        ${indicesAxis} -= sizeInConcatAxis[outputNumber - 1];
    }

    ${writeOutputData(outputs)}
  }`;
      return {
        ...metadata,
        getShaderSource,
        outputs: outputsTensorInfo,
        dispatchGroup: () => ({x: Math.ceil(inputSize / 64 /* workgroup size */)})
      };
    };

const createSplitProgramInfoLoader =
    (inputs: readonly TensorView[], attributes: SplitAttributes): ProgramInfoLoader => {
      const updatedAttributes = inputs.length === 1 ? attributes : createSplitAttributesFromInputs(inputs, attributes);
      const metadata:
          ProgramMetadata = {name: 'Split', inputTypes: [GpuDataType.default], cacheHint: updatedAttributes.cacheKey};
      return {...metadata, get: () => createSplitProgramInfo(metadata, [inputs[0]], attributes)};
    };

export const split = (context: ComputeContext, attributes: SplitAttributes): void => {
  validateInputs(context.inputs);
  context.compute(createSplitProgramInfoLoader(context.inputs, attributes), {inputs: [0]});
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
