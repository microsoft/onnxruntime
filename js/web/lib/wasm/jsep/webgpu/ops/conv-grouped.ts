// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {ProgramInfo, ProgramUniform} from '../types';

import {createTensorShapeVariables, getMaxComponents, inputVariable, outputVariable, ShaderHelper} from './common';
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
      const outputShapeInShader = [outputShape[0], outputShape[1], outputShape[2], outputShape[3] / components];

      const programUniforms: ProgramUniform[] = [
        {type: 'uint32', data: outputSize}, {type: 'int32', data: attributes.strides},
        {type: 'int32', data: attributes.pads}, ...createTensorShapeVariables(xShape),
        ...createTensorShapeVariables(wShape), ...createTensorShapeVariables(outputShapeInShader)
      ];
      const xNumber = (outputNumber - 1) * attributes.strides[1] + wShape[1];
      const getShaderSource = (shaderHelper: ShaderHelper) => {
        const output = outputVariable('output', inputs[0].dataType, outputShapeInShader.length, components);
        const {activationFunction, applyActivation} = getActivationSnippet(attributes, output.type.value);
        const x = inputVariable('x', inputs[0].dataType, xShape.length, components);
        const w = inputVariable('w', inputs[1].dataType, wShape.length, components);
        const inputVars = [x, w];
        if (hasBias) {
          inputVars.push(inputVariable('b', inputs[2].dataType, inputs[2].dims, components));
        }
        const processBias = hasBias ? 'value += b[output_channel];' : '';

        return `
  ${
            shaderHelper.registerUniform('output_size', 'u32')
                .registerUniform('strides', 'i32', 2)
                .registerUniform('pads', 'i32', 2)
                .declareVariables(...inputVars, output)}
  ${activationFunction}
  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.output_size')}
    let width0 = uniforms.output_shape[3];
    let output_channel = global_idx % width0;
    var index1 = global_idx / width0;
    let width1 = uniforms.output_shape[2] / ${outputNumber}u;
    let col = (index1 % width1) * ${outputNumber}u;
    index1 = index1 / width1;
    let row = index1 % uniforms.output_shape[1];
    let batch = index1 / uniforms.output_shape[1];

    let x_corner = vec2<i32>(i32(row), i32(col)) * uniforms.strides - uniforms.pads;

    var x_vals: array<${x.type.value}, ${xNumber}>;
    var values: array<${output.type.value}, ${outputNumber}>;
    let input_channel = output_channel;
    // Use constant instead of uniform can give better performance for w's height/width.
    for (var w_height: u32 = 0u; w_height < ${wShape[0]}; w_height++) {
      let x_height = x_corner.x + i32(w_height);
      if (x_height >= 0 || u32(x_height) < uniforms.x_shape[1]) {
        for (var i = 0; i < ${xNumber}; i++) {
          let x_width = x_corner.y + i;
          if (x_width >= 0 && u32(x_width) < uniforms.x_shape[2]) {
            x_vals[i] = ${x.get('batch', 'u32(x_height)', 'u32(x_width)', 'input_channel')};
          } else {
            x_vals[i] = ${x.type.value}(0);
          }
        }
        for (var w_width: u32 = 0u; w_width < ${wShape[1]}; w_width++) {
          let w_val = ${w.get('w_height', 'w_width', '0', 'output_channel')};
          for (var i = 0u; i < ${outputNumber}u; i++) {
            values[i] = fma(x_vals[i * ${attributes.strides[1]}u + w_width], w_val, values[i]);
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
        shaderCache: {
          hint: `${attributes.activationCacheKey};${components};${outputNumber};${xNumber};${wShape[0]};${wShape[1]}`,
          inputDependencies: hasBias ? ['rank', 'rank', 'type'] : ['rank', 'rank']
        },
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
          dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)},
          programUniforms
        }),
        getShaderSource,
      };
    };
