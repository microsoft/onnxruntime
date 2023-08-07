// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from '../types';

import {createIndicesHelper, ShaderHelper} from './common';

export const tileProgramMetadata = {
  name: 'Tile',
  inputTypes: [GpuDataType.default]
};

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length !== 2) {
    throw new Error('Tile requires 2 inputs.');
  }

  if (inputs[0].dataType !== DataType.float && inputs[0].dataType !== DataType.int32 &&
      inputs[0].dataType !== DataType.uint32) {
    throw new Error('Tile only support float, int32, and uint32 data types');
  }

  if (inputs[1].dataType !== DataType.int64) {
    throw new Error('Tile `repeats` input should be of int64 data type');
  }

  if (inputs[1].dims.length !== 1) {
    throw new Error('Tile `repeats` input should be 1-D');
  }

  const repeats: number[] = [];
  if (inputs[1].dims[0] > 0) {
    inputs[1].getBigInt64Array().forEach(v => repeats.push(Number(v)));
  }

  if (repeats.length !== inputs[0].dims.length) {
    throw new Error('Tile `repeats` input should have same number of elements as rank of input data tensor');
  }
};

const getOutputShape = (inputShape: readonly number[], repeats: readonly number[]): readonly number[] => {
  const outputShape: number[] = [];

  for (let i = 0; i < inputShape.length; ++i) {
    outputShape.push(inputShape[i] * repeats[i]);
  }

  return outputShape;
};

export const createTileProgramInfo =
    (tileProgramMetadata: ProgramMetadata, inputs: readonly TensorView[]): ProgramInfo => {
      // We currently only support 4-byte element tensors, so using f32 here is safe
      // TODO: support other data types for Tile
      const dataType = 'f32';
      const inputShape = inputs[0].dims;

      const repeats: number[] = [];
      if (inputs[1].dims[0] > 0) {
        inputs[1].getBigInt64Array().forEach(v => repeats.push(Number(v)));
      }

      const outputShape = getOutputShape(inputShape, repeats);
      const outputSize = ShapeUtil.size(outputShape);

      const inputIndicesHelper = createIndicesHelper('input', inputShape);
      const outputIndicesHelper = createIndicesHelper('output', outputShape);

      const isl = inputShape.length;
      const calculateInputIndexImpl = (): string => `
      fn calculateInputIndex(inputIndices: ${inputIndicesHelper.iType}, outputIndices: ${
          outputIndicesHelper.iType}) -> void {
        for (var i = 0; i < ${isl}; i++) {
          // TODO: IndicesHelper should offer uniform way to get/set indices for all ranks
          inputIndices${isl >= 2 ? '[i]' : ''} = (outputIndices${isl >= 2 ? '[i]' : ''} % inputShape[i]);
        }
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

    ${outputIndicesHelper.indicesVariableDeclaration('outputIndices')}
    ${outputIndicesHelper.o2iCall('global_idx', 'outputIndices')}
    ${inputIndicesHelper.indicesVariableDeclaration('inputIndices')}
    calculateInputIndexImpl(inputIndices, outputIndices);
    output[global_idx] = input[${inputIndicesHelper.i2oExpression('inputIndices')}];
  }`;

      return {
        ...tileProgramMetadata,
        outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
        getShaderSource,
        dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
      };
    };

export const tile = (context: ComputeContext): void => {
  validateInputs(context.inputs);
  const cacheHint = context.inputs.map(x => x.dims.toString()).join('_');
  context.compute(
      {...tileProgramMetadata, cacheHint, get: () => createTileProgramInfo(tileProgramMetadata, context.inputs)},
      {inputs: [0]});
};
