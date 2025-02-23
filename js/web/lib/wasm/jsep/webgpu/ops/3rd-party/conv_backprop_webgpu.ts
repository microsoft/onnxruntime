/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// sampled from [@tensorflow/tfjs] tfjs-backend-webgpu/src/conv_backprop_webgpu.ts

import { DataType } from '../../../../wasm-common';
import { LOG_DEBUG } from '../../../log';
import { TensorView } from '../../../tensor-view';
import { ShapeUtil } from '../../../util';
import { ProgramInfo, ProgramInputTensorInfoDependency, ProgramUniform } from '../../types';
import {
  createTensorShapeVariables,
  inputVariable,
  outputVariable,
  ShaderHelper,
  tensorTypeToWsglStorageType,
  UniformsArrayType,
  getMaxComponents,
} from '../common';
import { ConvTransposeAttributes } from '../conv-transpose';

export const createConvTranspose2DProgramInfo = (
  inputs: readonly TensorView[],
  attributes: ConvTransposeAttributes,
  squeezeOutputShapeFunction?: (shape: readonly number[]) => number[],
): ProgramInfo => {
  const hasBias = inputs.length > 2;
  const outputShape = attributes.outputShape;
  const isChannelsLast = attributes.format === 'NHWC';
  const group = attributes.group;
  const wShape = inputs[1].dims;
  const inputChannelsPerGroup = wShape[2] / group;
  const outputChannelsPerGroup = wShape[3];
  const aComponents = isChannelsLast ? getMaxComponents(inputChannelsPerGroup) : 1;
  const components = isChannelsLast ? getMaxComponents(outputChannelsPerGroup) : 1;
  const bComponents = isChannelsLast ? (outputChannelsPerGroup === 1 ? aComponents : components) : 1;
  const outputSize = ShapeUtil.size(outputShape) / components;
  const dispatch = [Math.ceil(outputSize / 64), 1, 1];
  LOG_DEBUG('verbose', () => `[conv2d_backprop_webgpu] dispatch = ${dispatch}`);

  const inputDependencies: ProgramInputTensorInfoDependency[] = ['rank', 'rank'];
  const strides = [attributes.strides[0], attributes.strides[1]];
  const filterDims = [attributes.kernelShape[isChannelsLast ? 1 : 2], attributes.kernelShape[isChannelsLast ? 2 : 3]];
  const dilations = [attributes.dilations[0], attributes.dilations[1]];
  const effectiveFilterDims = [
    filterDims[0] +
      (attributes.dilations[0] <= 1
        ? 0
        : (attributes.kernelShape[isChannelsLast ? 1 : 2] - 1) * (attributes.dilations[0] - 1)),
    filterDims[1] +
      (attributes.dilations[1] <= 1
        ? 0
        : (attributes.kernelShape[isChannelsLast ? 2 : 3] - 1) * (attributes.dilations[1] - 1)),
  ];
  const pads = [
    effectiveFilterDims[0] - 1 - Math.floor((attributes.pads[0] + attributes.pads[2]) / 2),
    effectiveFilterDims[1] - 1 - Math.floor((attributes.pads[1] + attributes.pads[3]) / 2),
  ];

  const programUniforms: ProgramUniform[] = [
    { type: DataType.uint32, data: outputSize },
    { type: DataType.uint32, data: strides },
    { type: DataType.uint32, data: filterDims },
    { type: DataType.uint32, data: dilations },
    { type: DataType.uint32, data: effectiveFilterDims },
    { type: DataType.int32, data: pads },
    { type: DataType.uint32, data: inputChannelsPerGroup },
    { type: DataType.uint32, data: outputChannelsPerGroup },
    ...createTensorShapeVariables(inputs[0].dims, inputs[1].dims),
  ];
  if (hasBias) {
    programUniforms.push(...createTensorShapeVariables(inputs[2].dims));
    inputDependencies.push('rank');
  }
  programUniforms.push(...createTensorShapeVariables(outputShape));

  const getShaderSource = (shaderHelper: ShaderHelper) => {
    const uniforms: UniformsArrayType = [
      { name: 'output_size', type: 'u32' },
      { name: 'strides', type: 'u32', length: strides.length },
      { name: 'filter_dims', type: 'u32', length: filterDims.length },
      { name: 'dilations', type: 'u32', length: filterDims.length },
      { name: 'effective_filter_dims', type: 'u32', length: effectiveFilterDims.length },
      { name: 'pads', type: 'i32', length: pads.length },
      { name: 'input_channels_per_group', type: 'u32' },
      { name: 'output_channels_per_group', type: 'u32' },
    ];
    const dataType = tensorTypeToWsglStorageType(inputs[0].dataType);
    const rowDim = isChannelsLast ? 1 : 2;
    const colDim = isChannelsLast ? 2 : 3;
    const channelDim = isChannelsLast ? 3 : 1;

    const w = inputVariable('W', inputs[1].dataType, inputs[1].dims.length, bComponents);
    const dy = inputVariable('Dy', inputs[0].dataType, inputs[0].dims.length, aComponents);
    const inputVariables = [dy, w];
    if (hasBias) {
      inputVariables.push(inputVariable('bias', inputs[2].dataType, [outputShape[channelDim]].length, components));
    }
    const output = outputVariable('result', inputs[0].dataType, outputShape.length, components);

    const calculateResult = (): string => {
      let calcStr = '';
      if (aComponents === 1) {
        calcStr += `
        let w_offset = ${w.indicesToOffset(`${w.type.indices}(u32(wRPerm), u32(wCPerm), inputChannel, wOutChannel)`)};
        let wValue = ${w.getByOffset(`w_offset / ${bComponents}`)};
        dotProd = dotProd + xValue * wValue;`;
      } else {
        if (outputChannelsPerGroup === 1) {
          calcStr += `
          let wValue = ${w.getByOffset(`${w.indicesToOffset(`${w.type.indices}(u32(wRPerm), u32(wCPerm), inputChannel, wOutChannel)`)} / ${bComponents}`)};
          dotProd = dotProd + dot(xValue, wValue);`;
        } else {
          for (let c = 0; c < aComponents; c++) {
            calcStr += `
            let wValue${c} = ${w.getByOffset(`${w.indicesToOffset(`${w.type.indices}(u32(wRPerm), u32(wCPerm), inputChannel + ${c}, wOutChannel)`)} / ${bComponents}`)};
            dotProd = dotProd + xValue[${c}] * wValue${c};`;
          }
        }
      }
      return calcStr;
    };
    const codeSnippet = `
            let outputIndices = ${output.offsetToIndices(`global_idx * ${components}`)};
            let batch = ${output.indicesGet('outputIndices', 0)};
            let d1 = ${output.indicesGet('outputIndices', channelDim)};
            let r = ${output.indicesGet('outputIndices', rowDim)};
            let c = ${output.indicesGet('outputIndices', colDim)};
            let dyCorner = vec2<i32>(i32(r), i32(c)) - uniforms.pads;
            let dyRCorner = dyCorner.x;
            let dyCCorner = dyCorner.y;
            let groupId = d1 / uniforms.output_channels_per_group;
            let wOutChannel = d1 - groupId * uniforms.output_channels_per_group;
            // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
            // ? = to be determined. : = across all values in that axis.
            var dotProd = ${output.type.value}(0.0);
            var wR: u32 = 0;
            if (uniforms.dilations.x == 1) {
              // Minimum wR >= 0 that satisfies (dyRCorner + wR) % (uniforms.strides.x) == 0
              wR = u32(((dyRCorner + i32(uniforms.strides.x) - 1) / i32(uniforms.strides.x)) * i32(uniforms.strides.x) - dyRCorner);
            }
            for (; wR < uniforms.effective_filter_dims.x; wR = wR + 1) {
              if (wR % uniforms.dilations.x != 0) {
                continue;
              }
              let dyR = (${dataType}(dyRCorner) + ${dataType}(wR)) / ${dataType}(uniforms.strides[0]);
              let wRPerm = uniforms.filter_dims.x - 1 - wR / uniforms.dilations.x;
              if (dyR < 0.0 || dyR >= ${dataType}(uniforms.Dy_shape[${rowDim}]) || fract(dyR) > 0.0 ||
                  wRPerm < 0) {
                continue;
              }
              let idyR: u32 = u32(dyR);
              var wC: u32 = 0;
              if (uniforms.dilations.y == 1) {
                // Minimum wC >= 0 that satisfies (dyCCorner + wC) % (uniforms.strides.y) == 0
                wC = u32(((dyCCorner + i32(uniforms.strides.y) - 1) / i32(uniforms.strides.y)) * i32(uniforms.strides.y) - dyCCorner);
              }

              for (; wC < uniforms.effective_filter_dims.y; wC = wC + 1) {
                if (wC % uniforms.dilations.y != 0) {
                  continue;
                }
                let dyC = (${dataType}(dyCCorner) + ${dataType}(wC)) / ${dataType}(uniforms.strides.y);
                let wCPerm = uniforms.filter_dims.y - 1 - wC / uniforms.dilations.y;
                if (dyC < 0.0 || dyC >= ${dataType}(uniforms.Dy_shape[${colDim}]) ||
                    fract(dyC) > 0.0 || wCPerm < 0) {
                  continue;
                }
                let idyC: u32 = u32(dyC);
                var inputChannel = groupId * uniforms.input_channels_per_group;
                for (var d2: u32 = 0; d2 < uniforms.input_channels_per_group; d2 = d2 + ${aComponents}) {
                  let xValue = ${
                    isChannelsLast
                      ? dy.getByOffset(
                          `${dy.indicesToOffset(`${dy.type.indices}(batch, idyR, idyC, inputChannel)`)} / ${aComponents}`,
                        )
                      : dy.get('batch', 'inputChannel', 'idyR', 'idyC')
                  };
                  ${calculateResult()}
                  inputChannel = inputChannel + ${aComponents};
                }
                wC = wC + uniforms.strides.y - 1;
              }
              wR = wR + uniforms.strides[0] - 1;
            }
            let value = dotProd${hasBias ? ` + bias[d1 / ${components}]` : ''};
            ${output.setByOffset('global_idx', 'value')};
          `;

    return `
    ${shaderHelper.registerUniforms(uniforms).declareVariables(...inputVariables, output)}
      ${shaderHelper.mainStart()}
      ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.output_size')};
    ${codeSnippet}}`;
  };

  return {
    name: 'ConvTranspose2D',
    shaderCache: {
      hint: `${attributes.cacheKey};${aComponents}${bComponents}${components}${outputChannelsPerGroup === 1}`,
      inputDependencies,
    },
    getRunData: () => ({
      dispatchGroup: { x: dispatch[0], y: dispatch[1], z: dispatch[2] },
      outputs: [
        {
          dims: squeezeOutputShapeFunction ? squeezeOutputShapeFunction(outputShape) : outputShape,
          dataType: inputs[0].dataType,
        },
      ],
      programUniforms,
    }),
    getShaderSource,
  };
};
