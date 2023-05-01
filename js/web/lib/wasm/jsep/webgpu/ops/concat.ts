// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {createIndicesHelper, IndicesHelper, ShaderHelper} from './common';

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

const createConcatProgramMetadata = (inputCount: number, cacheHint: string) =>
    ({name: 'Concat', inputTypes: Array(inputCount).fill(GpuDataType.default), cacheHint});

const calculateInputIndexImpl = (numberOfTensors: number): string => `
  fn calculateInputIndex(index: u32) -> u32 {
    for (var i: u32 = 0u; i < ${numberOfTensors}u; i += 1u ) {
      if (index < sizeInConcatAxis[i]) {
        return i;
      }
    }
    return ${numberOfTensors}u;
  }`;

const readBufferDataImpl = (indicesHelper: readonly IndicesHelper[], tensorRank: number, dataType: string) => {
  const numberOfTensors = indicesHelper.length;
  const codeLines: string[] = [];
  for (let i = 0; i < numberOfTensors; ++i) {
    const returnSnippet = `return input${i}[${indicesHelper[i].i2oExpression('indices', true)}];`;
    if (numberOfTensors === 1) {
      codeLines.push(returnSnippet);
    } else if (i === 0) {
      codeLines.push(`if (textureIndex == ${i}u) { ${returnSnippet} }`);
    } else if (i === numberOfTensors - 1) {
      codeLines.push(`else { ${returnSnippet} }`);
    } else {
      codeLines.push(`else if (textureIndex == ${i}) { ${returnSnippet} }`);
    }
  }
  return `
    fn readBufferData(textureIndex: u32, indices: ptr<function, ${indicesHelper[0].iType}>) -> ${dataType} {
      ${codeLines.join('\n')}
    }`;
};

const createConcatProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], axis: number, dataType = 'f32'): ProgramInfo => {
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
      const rank = outputShape.length;

      const sizeInConcatAxis = new Array<number>(inputs.length);
      const inputStorageBuffersDeclarations = new Array<string>(inputs.length);
      const inputIndicesHelpers = new Array<IndicesHelper>(inputs.length);

      let previousSum = 0;
      for (let i = 0; i < inputs.length; ++i) {
        previousSum += inputs[i].dims[adjustedAxis];
        sizeInConcatAxis[i] = previousSum;

        inputStorageBuffersDeclarations[i] =
            `@group(0) @binding(${i}) var<storage, read> input${i} : array<${dataType}>;`;

        inputIndicesHelpers[i] = createIndicesHelper(`input${i}`, inputs[i].dims);
      }

      const outputIndicesHelper = createIndicesHelper('output', outputShape);

      const indicesAxis = rank < 2 ? 'indices' : `indices[${adjustedAxis}]`;
      const getShaderSource = (shaderHelper: ShaderHelper) => `

  ${inputStorageBuffersDeclarations.join('\n')}
  @group(0) @binding(${inputs.length}) var<storage, read_write> output : array<${dataType}>;

  ${inputIndicesHelpers.map(i => i.i2oImpl).join('\n')}
  ${outputIndicesHelper.o2iImpl}

  const sizeInConcatAxis = array<u32, ${sizeInConcatAxis.length}>(${sizeInConcatAxis.map(i => `${i}u`).join(',')});
  ${calculateInputIndexImpl(sizeInConcatAxis.length)}
  ${readBufferDataImpl(inputIndicesHelpers, rank, dataType)}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

    ${outputIndicesHelper.indicesVariableDeclaration('indices')}
    ${outputIndicesHelper.o2iCall('global_idx', 'indices')}

    let textureIndex = calculateInputIndex(${indicesAxis});
    if (textureIndex != 0u) {
      ${indicesAxis} -= sizeInConcatAxis[textureIndex - 1u];
    }

    output[global_idx] = readBufferData(textureIndex, &indices);
  }`;
      return {
        ...metadata,
        outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
        getShaderSource,
        dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
      };
    };

const createConcatProgramInfoLoader =
    (inputs: readonly TensorView[], attributes: ConcatAttributes): ProgramInfoLoader => {
      const metadata = createConcatProgramMetadata(inputs.length, attributes.cacheKey);
      return {...metadata, get: () => createConcatProgramInfo(metadata, inputs, attributes.axis)};
    };

export const concat = (context: ComputeContext, attributes: ConcatAttributes): void => {
  validateInputs(context.inputs);
  context.compute(createConcatProgramInfoLoader(context.inputs, attributes));
};
