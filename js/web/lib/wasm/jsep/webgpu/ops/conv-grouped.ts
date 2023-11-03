// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {ProgramInfo} from '../types';

import {inputVariable, outputVariable, ShaderHelper} from './common';
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

      const {activationFunction, applyActivation} = getActivationSnippet(attributes);

      const isChannelLast = attributes.format === 'NHWC';
      const outputShape = calculateOutputShape(
          xShape, wShape, attributes.dilations, attributes.pads, attributes.strides, isChannelLast);
      const outputSize = ShapeUtil.size(outputShape);

      const output = outputVariable('output', inputs[0].dataType, outputShape);
      const x = inputVariable('x', inputs[0].dataType, xShape);
      const w = inputVariable('w', inputs[1].dataType, wShape);
      const inputVars = [x, w];
      if (hasBias) {
        inputVars.push(inputVariable('b', inputs[2].dataType, inputs[2].dims));
      }

      const getShaderSource = (shaderHelper: ShaderHelper) => `
  const strides: vec2<u32> = vec2(${attributes.strides[0]}u, ${attributes.strides[1]}u);
  const pads: vec2<u32> = vec2(${attributes.pads[0]}u, ${attributes.pads[1]}u);

  ${shaderHelper.declareVariables(...inputVars, output)}

  ${activationFunction}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

    let outputIndices = ${output.offsetToIndices('global_idx')};
    let batch: u32 = outputIndices[0];
    let output_channel: u32 = outputIndices[${isChannelLast ? 3 : 1}];
    let xRCCorner: vec2<u32> = vec2<u32>(outputIndices[${isChannelLast ? 1 : 2}], outputIndices[${
          isChannelLast ? 2 : 3}]) * strides - pads;
    let group_id: u32 = output_channel / ${outputChannelsPerGroup}u;

    var value: ${output.type.value} = ${output.type.value}(0);
    for (var wInChannel: u32 = 0u; wInChannel < ${wShape[1]}u; wInChannel++) {
      let input_channel = group_id * ${wShape[1]}u + wInChannel;
      for (var wHeight: u32 = 0u; wHeight < ${wShape[2]}u; wHeight++) {
        let xHeight = xRCCorner.x + wHeight * ${attributes.dilations[0]}u;

        if (xHeight < 0u || xHeight >= ${xShape[isChannelLast ? 1 : 2]}u) {
          continue;
        }

        for (var wWidth: u32 = 0u; wWidth < ${wShape[3]}u; wWidth++) {
          let xWidth = xRCCorner.y + wWidth * ${attributes.dilations[1]}u;
          if (xWidth < 0u || xWidth >= ${xShape[isChannelLast ? 2 : 3]}u) {
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
        shaderCache: {hint: attributes.cacheKey},
        getRunData: () => ({
          outputs: [{
            dims: squeezeOutputShapeFunction ? squeezeOutputShapeFunction(outputShape) : outputShape,
            dataType: inputs[0].dataType
          }],
          dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)},
        }),
        getShaderSource,
      };
    };
