// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo, ProgramInputTensorInfoDependency, ProgramUniform} from '../types';

import {createTensorShapeVariables, enableShapesUniforms, inputVariable, outputVariable, ShaderHelper} from './common';

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

      const indicesShape = inputs[1].dims;
      const indicesDataType = inputs[1].dataType;
      const axis = ShapeUtil.normalizeAxis(attributes.axis, inputRank);
      const axisDimLimit = inputShape[axis];

      const outputShape = indicesShape.slice(0);
      const outputSize = ShapeUtil.size(outputShape);
      const enableInputShapesUniforms = enableShapesUniforms(inputShape.length);
      const inputShapeOrRank = enableInputShapesUniforms ? inputShape.length : inputShape;
      const enableIndicesOutputShapesUniforms = enableShapesUniforms(indicesShape.length);
      const indicesOrOutputShapeOrRank = enableIndicesOutputShapesUniforms ? indicesShape.length : indicesShape;

      const input = inputVariable('input', inputOutputDataType, inputShapeOrRank);
      const indices = inputVariable('indicesInput', indicesDataType, indicesOrOutputShapeOrRank);
      const output = outputVariable('output', inputOutputDataType, indicesOrOutputShapeOrRank);


      const programUniforms: ProgramUniform[] =
          [{type: 'uint32', data: outputSize}, {type: 'int32', data: axisDimLimit}, {type: 'uint32', data: axis}];
      if (enableInputShapesUniforms) {
        programUniforms.push(...createTensorShapeVariables(inputShape));
      }
      if (enableIndicesOutputShapesUniforms) {
        programUniforms.push(...createTensorShapeVariables(indicesShape));
        programUniforms.push(...createTensorShapeVariables(outputShape));
      }
      const inputDependencies: ProgramInputTensorInfoDependency[] =
          [enableInputShapesUniforms ? 'rank' : 'dims', enableIndicesOutputShapesUniforms ? 'rank' : 'dims'];

      // int64 indices would be treated as little endian i32 with assumption they fall in i32 limits
      // That assumption is safe as it's not possible to allocate >2gb buffer for input tensor
      // Input data will be treated as u32 or two u32 for 8-byte tensors
      const getShaderSource = (shaderHelper: ShaderHelper) => `
      ${
          shaderHelper.registerUniform('outputSize', 'u32')
              .registerUniform('axisDimLimit', 'i32')
              .registerUniform('axis', 'u32')
              .declareVariables(input, indices, output)}
      ${shaderHelper.mainStart()}
      ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.outputSize')}

      let outputIndices = ${output.offsetToIndices('global_idx')};

      var idx = ${indices.getByIndices('outputIndices')};
      if (idx < 0) {
        idx = idx + uniforms.axisDimLimit;
      }
      var inputIndices = ${input.type.indices}(outputIndices);
      ${input.indicesSet('inputIndices', 'uniforms.axis', 'u32(idx)')};
      let value = ${input.getByIndices('inputIndices')};

      ${output.setByOffset('global_idx', 'value')};
  }`;

      return {
        name: 'GatherElements',
        shaderCache: {inputDependencies},
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
          dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)},
          programUniforms
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
