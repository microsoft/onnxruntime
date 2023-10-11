// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo} from '../types';

import {inputVariable, outputVariable, ShaderHelper} from './common';

export interface GatherElementsAttributes extends AttributeWithCacheKey {
  axis: number;
}

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length !== 2) {
    throw new Error('GatherElements requires 2 inputs.');
  }

  if (inputs[0].dims.length < 1) {
    throw new Error('GatherElements requires that the data input be rank >= 1.');
  }

  if (inputs[0].dims.length !== inputs[1].dims.length) {
    throw new Error(`GatherElements requires that the data input and
                     indices input tensors be of same rank.`);
  }
};

const createGatherElementsProgramInfo =
    (inputs: readonly TensorView[], attributes: GatherElementsAttributes): ProgramInfo => {
      const inputShape = inputs[0].dims;
      const inputOutputDataType = inputs[0].dataType;
      const inputRank = inputShape.length;
      const inputStrides = ShapeUtil.computeStrides(inputShape);
      const inputSize = ShapeUtil.size(inputShape);

      const indicesShape = inputs[1].dims;
      const indicesDataType = inputs[1].dataType;
      const indicesSize = ShapeUtil.size(indicesShape);

      const axis = ShapeUtil.normalizeAxis(attributes.axis, inputRank);
      const axisDimLimit = inputShape[axis];

      const outputShape = indicesShape.slice(0);
      const outputSize = ShapeUtil.size(outputShape);

      const input = inputVariable('input', inputOutputDataType, inputShape);
      const indices = inputVariable('indices', indicesDataType, [indicesSize]);
      const output = outputVariable('output', inputOutputDataType, outputShape);


      // int64 indices would be treated as little endian i32 with assumption they fall in i32 limits
      // That assumption is safe as it's not possible to allocate >2gb buffer for input tensor
      // Input data will be treated as u32 or two u32 for 8-byte tensors
      const getShaderSource = (shaderHelper: ShaderHelper) => `
      const inputStrides = array<u32, ${inputStrides.length}>(${inputStrides.map(i => `${i}u`).join(',')});
      ${shaderHelper.declareVariables(input, indices, output)}
      ${shaderHelper.mainStart()}
      ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

      let outputIndices = ${output.offsetToIndices('global_idx')};

      var idx = ${indices.getByOffset('global_idx')};
      if (idx < 0) {
        idx = idx + ${axisDimLimit};
      }

      var srcOffset = u32(0);

      for (var i = 0; i < ${inputShape.length}; i++) {
        if (i == ${axis}) {
          srcOffset +=  u32(idx) * inputStrides[i];
        } else {
          srcOffset += ${output.indicesGet('outputIndices', 'i')} * inputStrides[i];
        }
      }

      // Should never hit this with valid values in indices
      // This is a guard against malicious data in the indices input
      if (srcOffset < 0 || srcOffset >= ${inputSize}) {
        return;
      }

      output[global_idx] = input[srcOffset];
  }`;

      return {
        name: 'GatherElements',
        shaderCache: {hint: attributes.cacheKey},
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
          dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)}
        }),
        getShaderSource,
      };
    };

export const parseGatherElementsAttributes = (attributes: Record<string, unknown>): GatherElementsAttributes =>
    createAttributeWithCacheKey({axis: attributes.axis as number});

export const gatherElements = (context: ComputeContext, attributes: GatherElementsAttributes): void => {
  const inputs = context.inputs;
  validateInputs(inputs);
  context.compute(createGatherElementsProgramInfo(context.inputs, attributes));
};
