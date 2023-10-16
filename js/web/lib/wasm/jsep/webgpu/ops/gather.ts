// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo} from '../types';

import {inputVariable, outputVariable, ShaderHelper} from './common';

export interface GatherAttributes extends AttributeWithCacheKey {
  axis: number;
}

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length !== 2) {
    throw new Error('Gather requires 2 inputs.');
  }
};

const createGatherProgramInfo = (inputs: readonly TensorView[], attributes: GatherAttributes): ProgramInfo => {
  const inputShape = inputs[0].dims;
  const indicesShape = inputs[1].dims;

  const inputRank = inputShape.length;
  const axis = ShapeUtil.normalizeAxis(attributes.axis, inputRank);

  const outputShape = inputShape.slice(0);
  outputShape.splice(axis, 1, ...indicesShape);

  const axisDimLimit = inputShape[axis];
  const outputSize = ShapeUtil.size(outputShape);

  const data = inputVariable('data', inputs[0].dataType, inputs[0].dims);
  const indices = inputVariable('inputIndices', inputs[1].dataType, inputs[1].dims);
  const output = outputVariable('output', inputs[0].dataType, outputShape);
  const calcDataIndices = (): string => {
    const indicesRank = indicesShape.length;
    let calcStr = `var indicesIndices  = ${indices.type.indices}(0);`;
    for (let i = 0; i < indicesRank; i++) {
      calcStr += `${indicesRank > 1 ? `indicesIndices[${i}]` : 'indicesIndices'} = ${
          outputShape.length > 1 ? `outputIndices[${axis + i}]` : 'outputIndices'};`;
    }
    calcStr += `
        var idx = ${indices.getByIndices('indicesIndices')};
        if (idx < 0) {
          idx = idx + ${axisDimLimit};
        }
        var dataIndices = ${data.type.indices}(0);
      `;
    for (let i = 0, j = 0; i < inputRank; i++) {
      if (i === axis) {
        calcStr += `${inputRank > 1 ? `dataIndices[${i}]` : 'dataIndices'} = u32(idx);`;
        j += indicesRank;
      } else {
        calcStr += `${inputRank > 1 ? `dataIndices[${i}]` : 'dataIndices'} = ${
            outputShape.length > 1 ? `outputIndices[${j}]` : 'outputIndices'};`;
        j++;
      }
    }
    return calcStr;
  };

  const getShaderSource = (shaderHelper: ShaderHelper) => `
      ${shaderHelper.declareVariables(data, indices, output)}
      ${shaderHelper.mainStart()}
        ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
        let outputIndices = ${output.offsetToIndices('global_idx')};
        ${calcDataIndices()};
        let value = ${data.getByIndices('dataIndices')};
        ${output.setByOffset('global_idx', 'value')};
      }`;
  return {
    name: 'Gather',
    shaderCache: {hint: attributes.cacheKey},
    getRunData: () => ({
      outputs: [
        {dims: outputShape, dataType: inputs[0].dataType},
      ],
      dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)}
    }),
    getShaderSource,
  };
};

export const parseGatherAttributes = (attributes: Record<string, unknown>): GatherAttributes =>
    createAttributeWithCacheKey({axis: attributes.axis as number});

export const gather = (context: ComputeContext, attributes: GatherAttributes): void => {
  const inputs = context.inputs;
  validateInputs(inputs);
  context.compute(createGatherProgramInfo(context.inputs, attributes));
};
