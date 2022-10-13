// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../../../attribute-with-cache-key';
import {Graph} from '../../../graph';
import {OperatorInitialization} from '../../../operators';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGpuInferenceHandler} from '../inference-handler';
import {GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {createIndicesHelper, IndicesHelper, WORKGROUP_SIZE} from './common';

export interface ConcatAttributes extends AttributeWithCacheKey {
  readonly axis: number;
}

const validateInputs = (inputs: Tensor[]): void => {
  if (!inputs || inputs.length < 1) {
    throw new Error('too few inputs');
  }

  const inputType = inputs[0].type;
  const inputDimensionality = inputs[0].dims.length;

  // TODO: Support string concat
  if (inputType === 'string') {
    throw new Error('string tensor is not supported yet');
  }

  for (const input of inputs) {
    // make sure types of all inputs match
    if (input.type !== inputType) {
      throw new Error('input tensors should be one type');
    }

    // make sure the dimensionality of all inputs are the same
    if (input.dims.length !== inputDimensionality) {
      throw new Error('input tensors should have the same shape');
    }
  }
};

export const concat = async(
    inferenceHandler: WebGpuInferenceHandler, inputs: Tensor[], attributes: ConcatAttributes): Promise<Tensor[]> => {
  validateInputs(inputs);
  return inferenceHandler.run(createConcatProgramInfoLoader(inputs, attributes), inputs);
};

const createConcatProgramMetadata = (inputCount: number, cacheHint: string) =>
    ({name: 'Concat', inputTypes: Array(inputCount).fill(GpuDataType.default), cacheHint});

const createConcatProgramInfo =
    (metadata: ProgramMetadata, inputs: Tensor[], axis: number, dataType = 'f32'): ProgramInfo => {
      const inputShape = inputs[0].dims.slice();
      if (axis >= inputShape.length || axis < (-1 * inputShape.length)) {
        throw new Error('axis specified for concat doesn\'t match input dimensionality');
      }
      if (axis < 0) {
        axis = inputShape.length + axis;
      }
      // ensure all of the non-concatenated axes match each other
      // calculate the shape of the output tensor while we do that
      const outputShape = inputShape.slice(0);
      for (let i = 1; i < inputs.length; i++) {
        const dataNShape = inputs[i].dims.slice();
        for (let axisIndex = 0; axisIndex < inputShape.length; axisIndex++) {
          // add to the placeholder for computing output shape
          if (axisIndex === axis) {
            outputShape[axis] += dataNShape[axisIndex];
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
        previousSum += inputs[i].dims[axis];
        sizeInConcatAxis[i] = previousSum;

        inputStorageBuffersDeclarations[i] =
            `@group(0) @binding(${i}) var<storage, read> input${i} : array<${dataType}>;`;

        inputIndicesHelpers[i] = createIndicesHelper(`input${i}`, inputs[i].dims);
      }

      const outputIndicesHelper = createIndicesHelper('output', outputShape);

      const indicesAxis = rank < 2 ? 'indices' : `indices[${axis}]`;
      const shaderSource = `
  const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;

  ${inputStorageBuffersDeclarations.join('\n')}
  @group(0) @binding(${inputs.length}) var<storage, read_write> output : array<${dataType}>;

  ${inputIndicesHelpers.map(i => i.i2oImpl).join('\n')}
  ${outputIndicesHelper.o2iImpl}

  let sizeInConcatAxis = array<u32, ${sizeInConcatAxis.length}>(${sizeInConcatAxis.map(i => `${i}u`).join(',')});
  ${calculateInputIndexImpl(sizeInConcatAxis.length)}
  ${readBufferDataImpl(inputIndicesHelpers, rank, dataType)}

  @compute @workgroup_size(WORKGROUP_SIZE)
  fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {

    // Guard against out-of-bounds work group sizes
    if (global_id.x >= ${outputSize}u) {
      return;
    }

    ${outputIndicesHelper.indicesVariableDeclaration('indices')}
    ${outputIndicesHelper.o2iCall('global_id.x', 'indices')}

    let textureIndex = calculateInputIndex(${indicesAxis});
    if (textureIndex != 0u) {
      ${indicesAxis} -= sizeInConcatAxis[textureIndex - 1u];
    }

    output[global_id.x] = readBufferData(textureIndex, &indices);
  }`;
      return {
        ...metadata,
        outputs: [{dims: outputShape, type: inputs[0].type, gpuDataType: GpuDataType.default}],
        shaderSource,
        dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
      };
    };

const createConcatProgramInfoLoader = (inputs: Tensor[], attributes: ConcatAttributes): ProgramInfoLoader => {
  const metadata = createConcatProgramMetadata(inputs.length, attributes.cacheKey);
  return {...metadata, get: () => createConcatProgramInfo(metadata, inputs, attributes.axis)};
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

export const parseConcatAttributes: OperatorInitialization<ConcatAttributes> = (node: Graph.Node): ConcatAttributes =>
    createAttributeWithCacheKey({axis: node.attributes.getInt('axis')});
