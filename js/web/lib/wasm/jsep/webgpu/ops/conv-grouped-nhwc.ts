// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {inputVariable, outputVariable, ShaderHelper} from './common';
import {ConvAttributes} from './conv';
import {getActicationSnippet} from './fuse-utils';

const createConvNHWCProgramMetadata = (hasBias: boolean, cacheHint: string): ProgramMetadata => ({
  name: 'ConvNHWC',
  inputTypes: hasBias ? [GpuDataType.default, GpuDataType.default, GpuDataType.default] :
                        [GpuDataType.default, GpuDataType.default],
  cacheHint
});

const createConvNHWCProgramInfo =
    (inputs: readonly TensorView[], metadata: ProgramMetadata, attributes: ConvAttributes,
     outputShape: readonly number[],
     squeezeOutputShapeFunction?: (shape: readonly number[]) => number[]): ProgramInfo => {
      const hasBias = inputs.length > 2;
      const processBias = hasBias ? 'value += b[output_channel];' : '';
      const xShape = inputs[0].dims;  // [N, H, W, Ci]
      const wShape = inputs[1].dims;  // [kH, kW, Ci, Co]


      const {activationFunction, applyActivation} = getActicationSnippet(attributes);

      const isChannelLast = attributes.format === 'NHWC';
      const maxComponents = outputShape[3] % 4 === 0 && xShape[3] % 4 === 0 ? 4 : 1;
      const maxRowComponents = outputShape[2] % 2 === 0 ? 2 : 1;
      const outputSize = ShapeUtil.size(outputShape) / maxComponents / maxRowComponents;
      const output = outputVariable(
          'output', inputs[0].dataType,
          [outputShape[0], outputShape[1], outputShape[2], outputShape[3] / maxComponents], maxComponents);
      const x = inputVariable(
          'x', inputs[0].dataType, [xShape[0], xShape[1], xShape[2], xShape[3] / maxComponents], maxComponents);
      const w = inputVariable(
          'w', inputs[1].dataType, [wShape[0], wShape[1], wShape[2], wShape[3] / maxComponents], maxComponents);
      const inputVars = [x, w];
      if (hasBias) {
        inputVars.push(inputVariable('b', inputs[2].dataType, [inputs[2].dims[0] / maxComponents], maxComponents));
      }
      const xNumber = (maxRowComponents - 1) * attributes.strides[1] + wShape[1];
      const calcVecResult = (): string => {
        let calcStr = '';
        for (let i = 0; i < maxComponents; i++) {
          calcStr += `dotProd[i] = fma(${x.type.value}(xVal[${i}]), wVal${i}, dotProd[i]);`;
        }
        return calcStr;
      };
      const calcResult = (): string => {
        let calcStr = '';
        if (maxComponents === 1) {
          calcStr += `
          let wVal = ${w.get('wHeight', 'wWidth', 'wInChannel', 'output_channel')};
          for (var i = 0u; i < ${maxRowComponents}u; i++) {
            dotProd[i] = fma(xVals[i * strides[1] + wWidth], wVal, dotProd[i]);
          }`;
        } else {
          for (let i = 0; i < maxComponents; i++) {
            calcStr += `
            let wVal${i} = ${w.get('wHeight', 'wWidth', `wInChannel * ${maxComponents} + ${i}`, 'output_channel')};`;
          }
          calcStr += `
          for (var i = 0u; i < ${maxRowComponents}u; i++) {
            let xVal = xVals[i * strides[1] + wWidth];
            ${calcVecResult()}
          }`;
        }
        return calcStr;
      };

      const getShaderSource = (shaderHelper: ShaderHelper) => `
  const strides: vec2<u32> = vec2(${attributes.strides[0]}u, ${attributes.strides[1]}u);
  const pads: vec2<u32> = vec2(${attributes.pads[0]}u, ${attributes.pads[1]}u);

  ${shaderHelper.declareVariables(...inputVars, output)}

  ${activationFunction}
  fn readX(batch : u32, row : u32, col : u32, channel : u32) -> ${x.type.value} {
    var data = ${x.type.value}(0.0);
    if (col >=0 && col < ${x.shape[2]}) {
      data = ${x.get('batch', 'row', 'col', 'channel')};
    }
    return data;
  }
  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
    let output_channel = (global_idx % ${output.shape[3]}u);
    var index1 = global_idx / ${output.shape[3]}u;
    let width1 = ${output.shape[2] / maxRowComponents}u;
    let c = (index1 % width1) * ${maxRowComponents}u;
    index1 = index1 / width1;
    let r = index1 % ${output.shape[1]}u;
    let batch = index1 / ${output.shape[1]}u;

    let xRCCorner: vec2<u32> = vec2<u32>(r, c) * strides - pads;

    var dotProd: array<${output.type.value}, ${maxRowComponents}>;
    var xVals : array<${x.type.value}, ${xNumber}>;

      for (var wHeight: u32 = 0u; wHeight < ${wShape[0]}u; wHeight++) {
        let xHeight = xRCCorner.x + wHeight * ${attributes.dilations[0]}u;

        if (xHeight < 0u || xHeight >= ${xShape[isChannelLast ? 1 : 2]}u) {
          continue;
        }

        for (var wInChannel: u32 = 0u; wInChannel < ${x.shape[3]}u; wInChannel++) {
          for (var i = 0u; i < ${xNumber}u; i++) {
            xVals[i] = readX(batch, xHeight, xRCCorner.y + i, wInChannel);
          }
          for (var wWidth: u32 = 0u; wWidth < ${wShape[1]}u; wWidth++) {
            ${calcResult()}
          }
        }
    }
    for (var i = 0u; i < ${maxRowComponents}u; i++) {
        var value = dotProd[i];
        ${processBias}
        ${applyActivation}
        ${output.set('batch', 'r', 'c + i', 'output_channel', 'value')};
    };
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
export const createConvNHWCProgramInfoLoader =
    (inputs: readonly TensorView[], attributes: ConvAttributes, outputShape: readonly number[],
     squeezeOutputShapeFunction?: (shape: readonly number[]) => number[]): ProgramInfoLoader => {
      const metadata = createConvNHWCProgramMetadata(inputs.length > 2, attributes.cacheKey);
      return {
        ...metadata,
        get: () => createConvNHWCProgramInfo(inputs, metadata, attributes, outputShape, squeezeOutputShapeFunction)
      };
    };
