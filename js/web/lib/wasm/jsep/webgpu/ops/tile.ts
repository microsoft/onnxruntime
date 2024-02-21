// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {ComputeContext, ProgramInfo} from '../types';

import {inputVariable, outputVariable, ShaderHelper} from './common';

const getRepeats = (repeatsTensorView: TensorView): readonly number[] =>
    Array.from(repeatsTensorView.getBigInt64Array(), Number);


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

  const repeats: readonly number[] = getRepeats(inputs[1]);

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

export const createTileProgramInfo = (inputs: readonly TensorView[]): ProgramInfo => {
  const inputShape = inputs[0].dims;
  const repeats: readonly number[] = getRepeats(inputs[1]);
  const outputShape = getOutputShape(inputShape, repeats);
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
        let inputDimValue = ${output.indicesGet('outputIndices', 'i')}  % ${input.indicesGet('inputShape', 'i')};

        ${input.indicesSet('inputIndices', 'i', 'inputDimValue')}
      }
      ${output.setByOffset('global_idx', input.getByIndices('inputIndices'))}
    }`;

  return {
    name: 'Tile',
    shaderCache: {hint: `${repeats}`},
    getRunData: () => ({
      outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
      dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)},
    }),
    getShaderSource,
  };
};

export const tile = (context: ComputeContext): void => {
  validateInputs(context.inputs);
  context.compute(createTileProgramInfo(context.inputs), {inputs: [0]});
};
