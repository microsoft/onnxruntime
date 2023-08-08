// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {inputVariable, outputVariable, ShaderHelper} from './common';
import {calculateOutputShape, ConvAttributes} from './conv';
import {getActicationSnippet} from './fuse-utils';

const createGroupedConvProgramMetadata = (hasBias: boolean, cacheHint: string): ProgramMetadata => ({
  name: 'GroupedConv',
  inputTypes: hasBias ? [GpuDataType.default, GpuDataType.default, GpuDataType.default] :
                        [GpuDataType.default, GpuDataType.default],
  cacheHint
});

const createGroupedConvProgramInfo =
    (inputs: readonly TensorView[], metadata: ProgramMetadata, attributes: ConvAttributes,
     squeezeOutputShapeFunction?: (shape: readonly number[]) => number[]): ProgramInfo => {
      const hasBias = inputs.length > 2;
      const processBias = hasBias ? 'value += b[output_channel];' : '';
      const xShape = inputs[0].dims;
      const wShape = inputs[1].dims;
      const outputChannelsPerGroup = wShape[0] / attributes.group;

      const {activationFunction, applyActivation} = getActicationSnippet(attributes);

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
  const strides: vec2<i32> = vec2(${attributes.strides[0]}, ${attributes.strides[1]});
  const pads: vec2<i32> = vec2(${attributes.pads[0]}, ${attributes.pads[1]});

  ${shaderHelper.declareVariables(...inputVars, output)}

  ${activationFunction}
  ${output.impl('offsetToIndices')}
  ${x.impl('indicesToOffset', 'get')}
  ${w.impl('indicesToOffset', 'get')}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

    let outputIndices = ${output.offsetToIndices('global_idx')};
    let batch: i32 = outputIndices[0];
    let output_channel: i32 = outputIndices[${isChannelLast ? 3 : 1}];
    let xRCCorner: vec2<i32> = vec2<i32>(outputIndices[${isChannelLast ? 1 : 2}], outputIndices[${
          isChannelLast ? 2 : 3}]) * strides - pads;
    let group_id: i32 = output_channel / ${outputChannelsPerGroup};

    var value: ${output.type.value} = ${output.type.value}(0);
    for (var wInChannel: i32 = 0; wInChannel < ${wShape[1]}; wInChannel++) {
      let input_channel = group_id * ${wShape[1]} + wInChannel;
      for (var wHeight: i32 = 0; wHeight < ${wShape[2]}; wHeight++) {
        let xHeight = xRCCorner.x + wHeight * ${attributes.dilations[0]};

        if (xHeight < 0 || xHeight >= ${xShape[isChannelLast ? 1 : 2]}) {
          continue;
        }

        for (var wWidth: i32 = 0; wWidth < ${wShape[3]}; wWidth++) {
          let xWidth = xRCCorner.y + wWidth * ${attributes.dilations[1]};
          if (xWidth < 0 || xWidth >= ${xShape[isChannelLast ? 2 : 3]}) {
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
        ...metadata,
        outputs: [{
          dims: squeezeOutputShapeFunction ? squeezeOutputShapeFunction(outputShape) : outputShape,
          dataType: inputs[0].dataType,
          gpuDataType: GpuDataType.default
        }],
        getShaderSource,
        dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
      };
    };

/**
 * naive grouped conv implementation, supports 1d/2d conv
 * @param squeezeOutputShapeFunction - an optional function to squeeze the output shape, only used in conv1d
 */
export const createGroupedConvProgramInfoLoader =
    (inputs: readonly TensorView[], attributes: ConvAttributes,
     squeezeOutputShapeFunction?: (shape: readonly number[]) => number[]): ProgramInfoLoader => {
      const metadata = createGroupedConvProgramMetadata(inputs.length > 2, attributes.cacheKey);
      return {
        ...metadata,
        get: () => createGroupedConvProgramInfo(inputs, metadata, attributes, squeezeOutputShapeFunction)
      };
    };
