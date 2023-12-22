// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {ProgramInfo, ProgramInputTensorInfoDependency, ProgramUniform} from '../types';

import {createTensorShapeVariables, inputVariable, outputVariable, ShaderHelper, UniformsArrayType} from './common';
import {calculateOutputShape, ConvAttributes} from './conv';
import {getActivationSnippet} from './fuse-utils';

/**
 * naive grouped conv implementation, supports 1d/2d conv
 * @param squeezeOutputShapeFunction - an optional function to squeeze the output shape, only used in conv1d
 */
export const createGroupedConvProgramInfo =
    (inputs: readonly TensorView[], attributes: ConvAttributes,
     squeezeOutputShapeFunction?: (shape: readonly number[]) => number[]): ProgramInfo => {
      const hasBias = inputs.length > 2;
      const processBias = hasBias ? 'value += b[output_channel];' : '';
      const xShape = inputs[0].dims;
      const wShape = inputs[1].dims;
      const outputChannelsPerGroup = wShape[0] / attributes.group;

      const isChannelLast = attributes.format === 'NHWC';
      const outputShape = calculateOutputShape(
          xShape, wShape, attributes.dilations, attributes.pads, attributes.strides, isChannelLast);
      const outputSize = ShapeUtil.size(outputShape);

      const output = outputVariable('output', inputs[0].dataType, outputShape.length);
      const applyActivation = getActivationSnippet(attributes, output.type.value);
      const x = inputVariable('x', inputs[0].dataType, xShape.length);
      const w = inputVariable('w', inputs[1].dataType, wShape.length);
      const inputVars = [x, w];

      const programUniforms: ProgramUniform[] = [
        {type: 'uint32', data: outputSize}, {type: 'uint32', data: attributes.dilations},
        {type: 'uint32', data: [attributes.strides[0], attributes.strides[1]]},
        {type: 'uint32', data: [attributes.pads[0], attributes.pads[1]]}, {type: 'uint32', data: outputChannelsPerGroup}
      ];
      const uniforms: UniformsArrayType = [
        {name: 'outputSize', type: 'u32'}, {name: 'dilations', type: 'u32', length: attributes.dilations.length},
        {name: 'strides', type: 'u32', length: 2}, {name: 'pads', type: 'u32', length: 2},
        {name: 'outputChannelsPerGroup', type: 'u32'}
      ];
      if (attributes.activation === 'Clip') {
        programUniforms.push(
            {type: 'float32', data: attributes.clipMax!}, {type: 'float32', data: attributes.clipMin!});
        uniforms.push({name: 'clipMax', type: 'f32'}, {name: 'clipMin', type: 'f32'});
      }
      programUniforms.push(
          ...createTensorShapeVariables(xShape), ...createTensorShapeVariables(wShape),
          ...createTensorShapeVariables(outputShape));
      const inputDependencies: ProgramInputTensorInfoDependency[] = ['rank', 'rank'];
      if (hasBias) {
        inputVars.push(inputVariable('b', inputs[2].dataType, inputs[2].dims));
        programUniforms.push(...createTensorShapeVariables(inputs[2].dims));
        inputDependencies.push('rank');
      }
      programUniforms.push(...createTensorShapeVariables(outputShape));

      const getShaderSource = (shaderHelper: ShaderHelper) => `

  ${shaderHelper.registerUniforms(uniforms).declareVariables(...inputVars, output)}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.outputSize')}

    let outputIndices = ${output.offsetToIndices('global_idx')};
    let batch: u32 = outputIndices[0];
    let output_channel: u32 = outputIndices[${isChannelLast ? 3 : 1}];
    let xRCCorner: vec2<u32> = vec2<u32>(outputIndices[${isChannelLast ? 1 : 2}], outputIndices[${
          isChannelLast ? 2 : 3}]) * uniforms.strides - uniforms.pads;
    let group_id: u32 = output_channel / uniforms.outputChannelsPerGroup;

    var value: ${output.type.value} = ${output.type.value}(0);
    for (var wInChannel: u32 = 0u; wInChannel < uniforms.w_shape[1]; wInChannel++) {
      let input_channel = group_id * uniforms.w_shape[1] + wInChannel;
      for (var wHeight: u32 = 0u; wHeight < uniforms.w_shape[2]; wHeight++) {
        let xHeight = xRCCorner.x + wHeight * uniforms.dilations[0];

        if (xHeight < 0u || xHeight >= uniforms.x_shape[${isChannelLast ? 1 : 2}]) {
          continue;
        }

        for (var wWidth: u32 = 0u; wWidth < uniforms.w_shape[3]; wWidth++) {
          let xWidth = xRCCorner.y + wWidth * uniforms.dilations[1];
          if (xWidth < 0u || xWidth >= uniforms.x_shape[${isChannelLast ? 2 : 3}]) {
            continue;
          }

          let xVal = ${
          isChannelLast ? x.get('batch', 'xHeight', 'xWidth', 'input_channel') :
                          x.get('batch', 'input_channel', 'xHeight', 'xWidth')};
          let wVal = ${w.get('output_channel', 'wInChannel', 'wHeight', 'wWidth')};
          value += xVal*wVal;
        }
      }
    }
    ${processBias}
    ${applyActivation}
    ${output.setByOffset('global_idx', 'value')}
  }`;
      return {
        name: 'GroupedConv',
        shaderCache: {hint: attributes.cacheKey, inputDependencies},
        getRunData: () => ({
          outputs: [{
            dims: squeezeOutputShapeFunction ? squeezeOutputShapeFunction(outputShape) : outputShape,
            dataType: inputs[0].dataType
          }],
          dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)},
          programUniforms
        }),
        getShaderSource,
      };
    };
