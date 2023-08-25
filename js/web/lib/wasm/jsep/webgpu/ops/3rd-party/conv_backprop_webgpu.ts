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

import {LOG_DEBUG} from '../../../log';
import {TensorView} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {GpuDataType, ProgramInfo, ProgramMetadata} from '../../types';
import {inputVariable, outputVariable, ShaderHelper} from '../common';
import {ConvTransposeAttributes} from '../conv-transpose';

const createConvTranspose2DOpProgramShaderSource =
    (shaderHelper: ShaderHelper, inputs: readonly TensorView[], attributes: ConvTransposeAttributes,
     outputShape: readonly number[], hasBias: boolean, is1DimensionDispatch: boolean, isVec4 = false): string => {
      const isChannelsLast = attributes.format === 'NHWC';
      const rowDim = isChannelsLast ? 1 : 2;
      const colDim = isChannelsLast ? 2 : 3;
      const channelDim = isChannelsLast ? 3 : 1;
      const outputSize = ShapeUtil.size(outputShape);
      const workPerThread = isVec4 ? 2 : 1;
      const group = attributes.group;
      const wShape = inputs[1].dims;
      const inputChannelsPerGroup = wShape[0] / group;
      const outputChannelsPerGroup = wShape[1];

      let declareFunctions = `
  fn setOutputAtIndex(flatIndex : u32, value : ${isVec4 ? 'vec4<f32>' : 'f32'}) {
    result[flatIndex] = ${isVec4 ? 'vec4<f32>' : 'f32'}(value);
  }`;
      if (hasBias) {
        declareFunctions += `
    fn getBiasByOutputCoords(coords : vec4<u32>) -> ${isVec4 ? 'vec4<f32>' : 'f32'} {
      return bias[coords.${isChannelsLast ? 'w' : 'y'}${isVec4 ? '/ 4' : ''}];
    }`;
      }
      const components = isVec4 ? 4 : 1;
      const w = inputVariable('W', inputs[1].dataType, inputs[1].dims, components);
      const dy = inputVariable('Dy', inputs[0].dataType, inputs[0].dims, components);
      const inputVariables = [dy, w];
      if (hasBias) {
        inputVariables.push(inputVariable('bias', inputs[2].dataType, [outputShape[channelDim]], components));
      }
      const output = outputVariable('result', inputs[0].dataType, outputShape, components);
      const codeSnippet4 = `{
        let batch: u32 = ${is1DimensionDispatch ? 'global_id.z' : 'workgroup_id.z'} / outShape[1];
        let r = ${is1DimensionDispatch ? 'global_id.z' : 'workgroup_id.z'} % outShape[1];
        let c = ${is1DimensionDispatch ? 'global_id.y' : 'workgroup_id.y'} * ${workPerThread};
        let d1: u32 = ${is1DimensionDispatch ? 'global_id.x' : 'workgroup_id.x'} * 4;

        let dyCorner = vec2<i32>(i32(r), i32(c)) - vec2<i32>(pads);

        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
        // ? = to be determined. : = across all values in that axis.
        var dotProd: array<vec4<f32>, ${workPerThread}>;
        for (var i = 0; i < ${workPerThread}; i++) {
          dotProd[i] = vec4<f32>(0.0);
        }
        for (var wR: u32 = 0; wR < filterDims[0]; wR = wR + 1) {
          var dyR = (f32(dyCorner.x) + f32(wR)) / f32(strides.x);
          let wRPerm = filterDims[0] - 1 - wR;
          if (dyR < 0.0 || dyR >= f32(outBackprop[1]) ||
              fract(dyR) > 0.0 || wRPerm < 0) {
            continue;
          }
          let idyR: u32 = u32(dyR);

          for (var wC: u32 = 0; wC < filterDims[1]; wC = wC + 1) {
            let dyC = (f32(dyCorner.y) + f32(wC)) / f32(strides.y);
            let dyC2 = (f32(dyCorner.y) + 1.0 + f32(wC)) / f32(strides.y);
            let wCPerm = filterDims[1] - 1 - wC;
            if (wCPerm < 0) {
              continue;
            }
            var bDyCVal = true;
            var bDyCVal2 = true;
            if (dyC < 0.0 || dyC >= f32(outBackprop[2]) ||
                fract(dyC) > 0.0) {
              bDyCVal = false;
            }
            if (dyC2 < 0.0 || dyC2 >= f32(outBackprop[2]) ||
                fract(dyC2) > 0.0) {
              bDyCVal2 = false;
            }

            let idyC: u32 = u32(dyC);
            let idyC2: u32 = u32(dyC2);
            if (bDyCVal && bDyCVal2) {
              let d2Length = outBackprop[3];
              for (var d2 :u32 = 0; d2 < d2Length; d2 = d2 + 4) {
                let wValue0 = ${w.get('u32(wRPerm)', 'u32(wCPerm)', 'd1', 'd2')};
                let wValue1 = ${w.get('u32(wRPerm)', 'u32(wCPerm)', 'd1 + 1', 'd2')};
                let wValue2 = ${w.get('u32(wRPerm)', 'u32(wCPerm)', 'd1 + 2', 'd2')};
                let wValue3 = ${w.get('u32(wRPerm)', 'u32(wCPerm)', 'd1 + 3', 'd2')};

                var xValue = ${dy.get('batch', 'idyR', 'idyC', 'd2')};
                let tmpval = vec4<f32>(dot(xValue, wValue0),
                                      dot(xValue, wValue1),
                                      dot(xValue, wValue2),
                                      dot(xValue, wValue3));
                dotProd[0] = dotProd[0] + tmpval;

                xValue =  ${dy.get('batch', 'idyR', 'idyC2', 'd2')};

                dotProd[1] = dotProd[1] + vec4<f32>(dot(xValue, wValue0),
                                                    dot(xValue, wValue1),
                                                    dot(xValue, wValue2),
                                                    dot(xValue, wValue3));
              }
            } else if (bDyCVal) {
              let d2Length = outBackprop[${channelDim}];
              for (var d2: u32 = 0; d2 < d2Length; d2 = d2 + 4) {
                let wValue0 = ${w.get('u32(wRPerm)', 'u32(wCPerm)', 'd1', 'd2')};
                let wValue1 = ${w.get('u32(wRPerm)', 'u32(wCPerm)', 'd1 + 1', 'd2')};
                let wValue2 = ${w.get('u32(wRPerm)', 'u32(wCPerm)', 'd1 + 2', 'd2')};
                let wValue3 = ${w.get('u32(wRPerm)', 'u32(wCPerm)', 'd1 + 3', 'd2')};

                var xValue = ${dy.get('batch', 'idyR', 'idyC', 'd2')};
                let tmpval = vec4<f32>(dot(xValue, wValue0),
                                      dot(xValue, wValue1),
                                      dot(xValue, wValue2),
                                      dot(xValue, wValue3));
                dotProd[0] = dotProd[0] + tmpval;
              }
            } else if (bDyCVal2) {
              let d2Length = outBackprop[3];
              for (var d2: u32 = 0; d2 < d2Length; d2 = d2 + 4) {
                let wValue0 = ${w.get('u32(wRPerm)', 'u32(wCPerm)', 'd1', 'd2')};
                let wValue1 = ${w.get('u32(wRPerm)', 'u32(wCPerm)', 'd1 + 1', 'd2')};
                let wValue2 = ${w.get('u32(wRPerm)', 'u32(wCPerm)', 'd1 + 2', 'd2')};
                let wValue3 = ${w.get('u32(wRPerm)', 'u32(wCPerm)', 'd1 + 3', 'd2')};

                var xValue = ${dy.get('batch', 'idyR', 'idyC2', 'd2')};
                let tmpval = vec4<f32>(dot(xValue, wValue0),
                                      dot(xValue, wValue1),
                                      dot(xValue, wValue2),
                                      dot(xValue, wValue3));
                dotProd[1] = dotProd[1] + tmpval;
              }
            }
          }
        }

        for (var i: u32 = 0; i < ${workPerThread}; i = i + 1) {
          let value = dotProd[i] + ${hasBias ? 'bias[c+i]' : '0.0'};
          ${output.set('batch', 'r', 'c + i', 'd1', 'value')};
        }
      }`;
      const codeSnippet = `
          let outputIndices = ${output.offsetToIndices('global_idx')};
          let batch = ${output.indicesGet('outputIndices', 0)};
          let d1 = ${output.indicesGet('outputIndices', channelDim)};
          let r = ${output.indicesGet('outputIndices', rowDim)};
          let c = ${output.indicesGet('outputIndices', colDim)};
          let dyCorner = vec2<i32>(i32(r), i32(c)) - pads;
          let dyRCorner = dyCorner.x;
          let dyCCorner = dyCorner.y;
          let groupId = d1 / ${outputChannelsPerGroup};
          let wOutChannel = d1 - groupId * ${outputChannelsPerGroup};
          // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
          // ? = to be determined. : = across all values in that axis.
          var dotProd = 0.0;
          for (var wR: u32 = 0; wR < effectiveFilterDims.x; wR = wR + 1) {
            if (wR % dilations.x != 0) {
              continue;
            }
            let dyR = (f32(dyRCorner) + f32(wR)) / f32(strides[0]);
            let wRPerm = filterDims.x - 1 - wR / dilations.x;
            if (dyR < 0.0 || dyR >= f32(outBackprop[${rowDim}]) || fract(dyR) > 0.0 ||
                wRPerm < 0) {
              continue;
            }
            let idyR: u32 = u32(dyR);

            for (var wC: u32 = 0; wC < effectiveFilterDims.y; wC = wC + 1) {
              if (wC % dilations.y != 0) {
                continue;
              }
              let dyC = (f32(dyCCorner) + f32(wC)) / f32(strides.y);
              let wCPerm = filterDims.y - 1 - wC / dilations.y;
              if (dyC < 0.0 || dyC >= f32(outBackprop[${colDim}]) ||
                  fract(dyC) > 0.0 || wCPerm < 0) {
                continue;
              }
              let idyC: u32 = u32(dyC);

              for (var d2: u32 = 0; d2 < ${inputChannelsPerGroup}; d2 = d2 + 1) {
                let inputChannel = groupId * ${inputChannelsPerGroup} + d2;
                let xValue = ${
          isChannelsLast ? dy.get('batch', 'idyR', 'idyC', 'inputChannel') :
                           dy.get('batch', 'inputChannel', 'idyR', 'idyC')};
                let wValue = ${w.get('inputChannel', 'wOutChannel', 'u32(wRPerm)', 'u32(wCPerm)')};
                dotProd = dotProd + xValue * wValue;
              }
            }
          }
          let value = dotProd + ${hasBias ? 'bias[d1]' : '0.0'};
          ${output.setByOffset('global_idx', 'value')};
        `;

      return `
  ${shaderHelper.declareVariables(...inputVariables, output)}
  ${declareFunctions}
  const outShape : vec4<u32> = vec4<u32>(${outputShape.join(',')});
  const outBackprop : vec4<u32> = vec4<u32>(${inputs[0].dims.join(',')});
  const strides : vec2<u32> = vec2<u32>(${attributes.strides[0]}, ${attributes.strides[1]});
  const filterDims : vec2<u32> = vec2<u32>(${attributes.kernelShape[isChannelsLast ? 1 : 2]}, ${
          attributes.kernelShape[isChannelsLast ? 2 : 3]});
  const dilations : vec2<u32> = vec2<u32>(${attributes.dilations[0]}, ${attributes.dilations[1]});
  const effectiveFilterDims : vec2<u32> = filterDims + vec2<u32>(
          ${
          attributes.dilations[0] <= 1 ?
              0 :
              (attributes.kernelShape[isChannelsLast ? 1 : 2] - 1) * (attributes.dilations[0] - 1)},
          ${
          attributes.dilations[1] <= 1 ?
              0 :
              (attributes.kernelShape[isChannelsLast ? 2 : 3] - 1) * (attributes.dilations[1] - 1)});
  const pads : vec2<i32> = vec2<i32>(i32(effectiveFilterDims[0]) - 1 - (${attributes.pads[0] + attributes.pads[2]})/2,
                                     i32(effectiveFilterDims[1]) - 1 - (${attributes.pads[1] + attributes.pads[3]})/2);
    ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)};
  ${isVec4 ? codeSnippet4 : codeSnippet}}`;
    };

export const createConvTranspose2DProgramInfo =
    (inputs: readonly TensorView[], metadata: ProgramMetadata, attributes: ConvTransposeAttributes,
     squeezeOutputShapeFunction?: (shape: readonly number[]) => number[]): ProgramInfo => {
      const hasBias = inputs.length > 2;
      // const isChannelsLast = attributes.format === 'NHWC';
      const outputShape = attributes.outputShape;
      const outputSize = ShapeUtil.size(outputShape);

      // const inChannels = inputs[0].dims[isChannelsLast ? 3 : 1];
      // TODO Enable isVec4 for performance
      // Disabled due to weight matrix layout issue
      // const isVec4 = attributes.group === 1 && isChannelsLast && inChannels % 4 === 0 && outChannels % 4 === 0;
      const dispatch = [
        Math.ceil(outputSize / 64),
        1,
        1,
      ];
      LOG_DEBUG('verbose', () => `[conv2d_backprop_webgpu] dispatch = ${dispatch}`);

      return {
        ...metadata,
        outputs: [{
          dims: squeezeOutputShapeFunction ? squeezeOutputShapeFunction(outputShape) : outputShape,
          dataType: inputs[0].dataType,
          gpuDataType: GpuDataType.default
        }],
        dispatchGroup: () => ({x: dispatch[0], y: dispatch[1], z: dispatch[2]}),
        getShaderSource: (shaderHelper: ShaderHelper) => createConvTranspose2DOpProgramShaderSource(
            shaderHelper, inputs, attributes, outputShape, hasBias, dispatch[1] === 1 && dispatch[2] === 1),
      };
    };
