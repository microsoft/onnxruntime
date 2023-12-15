// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {ProgramInfo} from '../types';

import {getMaxComponents, inputVariable, outputVariable, ShaderHelper} from './common';
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

      const output = outputVariable('output', inputs[0].dataType, outputShape);
      const {activationFunction, applyActivation} = getActivationSnippet(attributes, output.type.value);
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

export const createGroupedConvVectorizeProgramInfo =
    (inputs: readonly TensorView[], attributes: ConvAttributes, outputShape: readonly number[]): ProgramInfo => {
      const hasBias = inputs.length > 2;
      const components = getMaxComponents(outputShape[3]);
      const outputNumber = getMaxComponents(outputShape[2]);
      const outputSize = ShapeUtil.size(outputShape) / components / outputNumber;
      const xShape = [inputs[0].dims[0], inputs[0].dims[1], inputs[0].dims[2], inputs[0].dims[3] / components];
      const wShape = [inputs[1].dims[0], inputs[1].dims[1], inputs[1].dims[2], inputs[1].dims[3] / components];

      const getShaderSource = (shaderHelper: ShaderHelper) => {
        const output = outputVariable(
            'output', inputs[0].dataType, [outputShape[0], outputShape[1], outputShape[2], outputShape[3] / components],
            components);
        const {activationFunction, applyActivation} = getActivationSnippet(attributes, output.type.value);
        const x = inputVariable('x', inputs[0].dataType, xShape, components);
        const w = inputVariable('w', inputs[1].dataType, wShape, components);
        const inputVars = [x, w];
        if (hasBias) {
          inputVars.push(inputVariable('b', inputs[2].dataType, inputs[2].dims, components));
        }
        const processBias = hasBias ? 'value += b[output_channel];' : '';
        const xNumber = (outputNumber - 1) * attributes.strides[1] + wShape[1];

        return `
  const strides: vec2<i32> = vec2(${attributes.strides[0]}, ${attributes.strides[1]});
  const pads: vec2<i32> = vec2(${attributes.pads[0]}, ${attributes.pads[1]});
  ${shaderHelper.declareVariables(...inputVars, output)}
  ${activationFunction}
  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
    let width0 = ${outputShape[3]}u / ${components}u;
    let output_channel = global_idx % width0;
    var index1 = global_idx / width0;
    let width1 = ${outputShape[2]}u / ${outputNumber}u;
    let col = (index1 % width1) * ${outputNumber}u;
    index1 = index1 / width1;
    let row = index1 % ${outputShape[1]}u;
    let batch = index1 / ${outputShape[1]}u;

    let xRCCorner = vec2<i32>(i32(row), i32(col)) * strides - pads;

    var xVals: array<${x.type.value}, ${xNumber}>;
    var values: array<${output.type.value}, ${outputNumber}>;
    let input_channel = output_channel;
      for (var wHeight: u32 = 0u; wHeight < ${wShape[0]}u; wHeight++) {
        let xHeight = xRCCorner.x + i32(wHeight);
        if (xHeight >= 0 || xHeight < ${xShape[1]}) {
          for (var i = 0; i < ${xNumber}; i++) {
            let xWidth = xRCCorner.y + i;
            if (xWidth >= 0 && xWidth < ${xShape[2]}) {
              xVals[i] = ${x.get('batch', 'u32(xHeight)', 'u32(xWidth)', 'input_channel')};
            } else {
              xVals[i] = ${x.type.value}(0);
            }
          }
          for (var wWidth: u32 = 0u; wWidth < ${wShape[1]}u; wWidth++) {
            let wVal = ${w.get('wHeight', 'wWidth', '0', 'output_channel')};
            for (var i = 0u; i < ${outputNumber}u; i++) {
              values[i] = fma(xVals[i * ${attributes.strides[1]}u + wWidth], wVal, values[i]);
            }
          }
        }
      }

    for (var i = 0u; i < ${outputNumber}u; i++) {
        var value = values[i];
        ${processBias}
        ${applyActivation}
        ${output.set('batch', 'row', 'col + i', 'output_channel', 'value')};
    }
  }`;
      };

      return {
        name: 'GroupedConv-Vectorize',
        shaderCache: {hint: attributes.cacheKey},
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
          dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)},
        }),
        getShaderSource,
      };
    };
