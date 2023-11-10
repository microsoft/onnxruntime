// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata, TensorInfo} from '../types';

import {createIndicesHelper, IndicesHelper, ShaderHelper} from './common';

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
fn calculateOutputIndex(index: u32) -> u32 {
    for (var i: u32 = 0u; i < ${numberOfTensors}u; i += 1u ) {
    if (index < sizeInConcatAxis[i]) {
        return i;
    }
    }
    return ${numberOfTensors}u;
}`;
const writeBufferDataImpl = (indicesHelper: readonly IndicesHelper[]) => {
  const numberOfTensors = indicesHelper.length;
  const codeLines: string[] = [];
  for (let i = 0; i < numberOfTensors; ++i) {
    const returnSnippet = `output${i}[${indicesHelper[i].i2oExpression('indices', true)}] = input[global_idx];`;
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
      fn writeBufferData(outputNumber: u32, indices: ptr<function, ${indicesHelper[0].iType}>, global_idx: u32) {
        ${codeLines.join('\n')}
      }`;
};

const createSplitProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: SplitAttributes, dataType = 'f32'):
        ProgramInfo => {
          const inputShape = inputs[0].dims;
          const inputSize = ShapeUtil.size(inputShape);
          const rank = inputShape.length;
          const axis = attributes.axis;
          const adjustedAxis = (axis < 0) ? inputShape.length + axis : axis;
          const outputStorageBuffersDeclarations = new Array<string>(attributes.numOutputs);
          const outputIndicesHelpers = new Array<IndicesHelper>(attributes.numOutputs);
          const inputIndicesHelper = createIndicesHelper('input', inputShape);
          const sizeInConcatAxis = new Array<number>(attributes.numOutputs);
          const outputs: TensorInfo[] = [];
          const outputShapes: number[][] = [];
          let previousSum = 0;
          for (let i = 0; i < attributes.numOutputs; i++) {
            previousSum += attributes.splitSizes[i];
            sizeInConcatAxis[i] = previousSum;
            outputStorageBuffersDeclarations[i] =
                `@group(0) @binding(${i + 1}) var<storage, read_write> output${i} : array<${dataType}>;`;
            const outputShape = inputShape.slice();
            outputShape[attributes.axis] = attributes.splitSizes[i];
            outputShapes.push(outputShape);
            outputIndicesHelpers[i] = createIndicesHelper(`output${i}`, outputShapes[i]);
            outputs.push({dims: outputShapes[i], dataType: inputs[0].dataType, gpuDataType: GpuDataType.default});
          }
          const indicesAxis = rank < 2 ? 'indices' : `indices[${adjustedAxis}]`;
          const getShaderSource = (shaderHelper: ShaderHelper) => `
  @group(0) @binding(0) var<storage, read> input : array<${dataType}>;
  ${outputStorageBuffersDeclarations.join('\n')}
  ${inputIndicesHelper.o2iImpl}
  ${outputIndicesHelpers.map(o => o.i2oImpl).join('\n')}
  const sizeInConcatAxis = array<u32, ${sizeInConcatAxis.length}>(${sizeInConcatAxis.map(i => `${i}u`).join(',')});
  ${calculateOutputIndexImpl(sizeInConcatAxis.length)}
  ${writeBufferDataImpl(outputIndicesHelpers)}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(inputSize)}

    ${inputIndicesHelper.indicesVariableDeclaration('indices')}
    ${inputIndicesHelper.o2iCall('global_idx', 'indices')}
    let outputNumber = calculateOutputIndex(${indicesAxis});
    if (outputNumber != 0) {
        ${indicesAxis} -= sizeInConcatAxis[outputNumber - 1u];
    }
    writeBufferData(outputNumber, &indices, global_idx);
  }`;
          return {
            ...metadata,
            getShaderSource,
            outputs,
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
