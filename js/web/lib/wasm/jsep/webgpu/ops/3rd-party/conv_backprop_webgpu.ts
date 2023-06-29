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
import {createIndicesHelper, ShaderHelper} from '../common';
import {ConvTransposeAttributes} from '../conv-transpose';

const createConvTranspose2DOpProgramShaderSource =
    (shaderHelper: ShaderHelper, inputs: readonly TensorView[], attributes: ConvTransposeAttributes,
     outputShape: readonly number[], hasBias: boolean, elementsPerThread: readonly number[]): string => {
      const isChannelsLast = attributes.format === 'NHWC';
      const rowDim = isChannelsLast ? 1 : 2;
      const colDim = isChannelsLast ? 2 : 3;
      const channelDim = isChannelsLast ? 3 : 1;
      const outputSize = ShapeUtil.size(outputShape);
      const outChannels = outputShape[isChannelsLast ? 3 : 1];
      const inChannels = inputs[0].dims[isChannelsLast ? 3 : 1];
      const isVec4 = inChannels % 4 === 0 && outChannels % 4 === 0;
      const workPerThread = isVec4 ? 2 : 1;

      const innerElementSize = isVec4 ? (isChannelsLast && inChannels % 4 !== 0 ? 3 : 4) : elementsPerThread[0];

      const declareInputs = [
        `@group(0) @binding(0) var<storage, read> Dy: array<${
            isVec4 && innerElementSize === 4 ? 'vec4<f32>' : 'f32'}>;`,
        `@group(0) @binding(1) var<storage, read> W: array<${isVec4 ? 'vec4<f32>' : 'f32'}>;`
      ];
      let declareFunctions = `
  fn setOutputAtIndex(flatIndex : u32, value : ${isVec4 ? 'vec4<f32>' : 'f32'}) {
    result[flatIndex] = ${isVec4 ? 'vec4<f32>' : 'f32'}(value);
  }`;
      if (hasBias) {
        declareInputs.push(`@group(0) @binding(2) var<storage, read> bias: array<${isVec4 ? 'vec4<f32>' : 'f32'}>;`);
        declareFunctions += `
    fn getBiasByOutputCoords(coords : vec4<u32>) -> ${isVec4 ? 'vec4<f32>' : 'f32'} {
      return bias[coords.${isChannelsLast ? 'w' : 'y'}${isVec4 ? '/ 4' : ''}];
    }`;
      }
      const wIndicesHelper = createIndicesHelper('W', inputs[1].dims);
      const dyIndicesHelper = createIndicesHelper('Dy', inputs[0].dims);
      const outputIndicesHelper = createIndicesHelper('result', outputShape);
      const codeSnippet4 = `{
        let batch: u32 = global_id.z / outShape[1];
        let r = global_id.z % outShape[1];
        let c = global_id.y * ${workPerThread};
        let d1: u32 = global_id.x * 4;

        let dyCorner = vec2<i32>(i32(r), i32(c)) - vec2<i32>(pads);

        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
        // ? = to be determined. : = across all values in that axis.
        var dotProd: array<vec4<f32>, ${workPerThread}>;
        for (var i = 0; i < ${workPerThread}; i++) {
          dotProd[i] = vec4<f32>(0.0);
        }
        for (var wR: u32 = 0; wR < filterDims[0]; wR = wR + 1) {
          var dyR = f32(dyCorner.x + wR) / f32(strides.x);
          let wRPerm: u32= filterDims[0] - 1 - wR;
          if (dyR < 0.0 || dyR >= f32(outBackprop[1]) ||
              fract(dyR) > 0.0) {
            continue;
          }
          let idyR: u32 = u32(dyR);

          for (var wC: u32 = 0; wC < filterDims[1]; wC = wC + 1) {
            let dyC = f32(dyCorner.y + wC) / f32(strides.y);
            let dyC2 = f32(dyCorner.y + 1 + wC) / f32(strides.y);
            let wCPerm: u32 = filterDims[1] - 1 - wC;
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
                ${
          wIndicesHelper.indicesVariableDeclaration(
              'wIndices0',
              [
                'd2', 'd1', 'wRPerm', 'wCPerm'
              ])};
                ${
          wIndicesHelper.indicesVariableDeclaration(
              'wIndices1',
              [
                'd2', 'd1+1', 'wRPerm', 'wCPerm'
              ])};
                ${
          wIndicesHelper.indicesVariableDeclaration(
              'wIndices2',
              [
                'd2', 'd1+2', 'wRPerm', 'wCPerm'
              ])};
                ${
          wIndicesHelper.indicesVariableDeclaration(
              'wIndices3',
              [
                'd2', 'd1+3', 'wRPerm', 'wCPerm'
              ])};
                let wValue0 = W[${wIndicesHelper.i2oExpression('wIndices0')}];
                let wValue1 = W[${wIndicesHelper.i2oExpression('wIndices1')}];
                let wValue2 = W[${wIndicesHelper.i2oExpression('wIndices2')}];
                let wValue3 = W[${wIndicesHelper.i2oExpression('wIndices3')}];
                ${
          dyIndicesHelper.indicesVariableDeclaration(
              'dyIndices',
              isChannelsLast ? ['batch', 'idyR', 'idyC', 'd2'] :
                               [
                                 'batch', 'd2', 'idyR', 'idyC'
                               ])};
                var xValue =  Dy[${dyIndicesHelper.i2oExpression('dyIndices')}];
                let tmpval = vec4<f32>(xValue * wValue0,
                                      xValue * wValue1,
                                      xValue * wValue2,
                                      xValue * wValue3);
                dotProd[0] = dotProd[0] + tmpval;

                ${
          dyIndicesHelper.indicesVariableDeclaration(
              'dyIndices2',
              isChannelsLast ? ['batch', 'idyR', 'idyC2', 'd2'] :
                               [
                                 'batch', 'd2', 'idyR', 'idyC2'
                               ])};
                xValue =  Dy[${dyIndicesHelper.i2oExpression('dyIndices')}];

                dotProd[1] = dotProd[1] + vec4<f32>(xValue * wValue0,
                                                    xValue * wValue1,
                                                    xValue * wValue2,
                                                    xValue * wValue3);
              }
            } else if (bDyCVal) {
              let d2Length = outBackprop[3];
              for (var d2: u32 = 0; d2 < d2Length; d2 = d2 + 4) {
                ${
          wIndicesHelper.indicesVariableDeclaration(
              'wIndices0',
              [
                'd2', 'd1', 'wRPerm', 'wCPerm'
              ])};
                ${
          wIndicesHelper.indicesVariableDeclaration(
              'wIndices1',
              [
                'd2', 'd1+1', 'wRPerm', 'wCPerm'
              ])};
                ${
          wIndicesHelper.indicesVariableDeclaration(
              'wIndices2',
              [
                'd2', 'd1+2', 'wRPerm', 'wCPerm'
              ])};
                ${
          wIndicesHelper.indicesVariableDeclaration(
              'wIndices3',
              [
                'd2', 'd1+3', 'wRPerm', 'wCPerm'
              ])};
                let wValue0 = W[${wIndicesHelper.i2oExpression('wIndices0')}];
                let wValue1 = W[${wIndicesHelper.i2oExpression('wIndices1')}];
                let wValue2 = W[${wIndicesHelper.i2oExpression('wIndices2')}];
                let wValue3 = W[${wIndicesHelper.i2oExpression('wIndices3')}];
                ${
          dyIndicesHelper.indicesVariableDeclaration(
              'dyIndices',
              isChannelsLast ? ['batch', 'idyR', 'idyC', 'd2'] :
                               [
                                 'batch', 'd2', 'idyR', 'idyC'
                               ])};
                var xValue =  Dy[${dyIndicesHelper.i2oExpression('dyIndices')}];
                let tmpval = vec4<f32>(xValue * wValue0,
                                      xValue * wValue1,
                                      xValue * wValue2,
                                      xValue * wValue3);
                dotProd[0] = dotProd[0] + tmpval;
              }
            } else if (bDyCVal2) {
              let d2Length = outBackprop[3];
              for (var d2: u32 = 0; d2 < d2Length; d2 = d2 + 4) {
                ${
          wIndicesHelper.indicesVariableDeclaration(
              'wIndices0',
              [
                'd2', 'd1', 'wRPerm', 'wCPerm'
              ])};
                ${
          wIndicesHelper.indicesVariableDeclaration(
              'wIndices1',
              [
                'd2', 'd1+1', 'wRPerm', 'wCPerm'
              ])};
                ${
          wIndicesHelper.indicesVariableDeclaration(
              'wIndices2',
              [
                'd2', 'd1+2', 'wRPerm', 'wCPerm'
              ])};
                ${
          wIndicesHelper.indicesVariableDeclaration(
              'wIndices3',
              [
                'd2', 'd1+3', 'wRPerm', 'wCPerm'
              ])};
                let wValue0 = W[${wIndicesHelper.i2oExpression('wIndices0')}];
                let wValue1 = W[${wIndicesHelper.i2oExpression('wIndices1')}];
                let wValue2 = W[${wIndicesHelper.i2oExpression('wIndices2')}];
                let wValue3 = W[${wIndicesHelper.i2oExpression('wIndices3')}];
                ${
          dyIndicesHelper.indicesVariableDeclaration(
              'dyIndices',
              isChannelsLast ? ['batch', 'idyR', 'idyC', 'd2'] :
                               [
                                 'batch', 'd2', 'idyR', 'idyC'
                               ])};
                var xValue =  Dy[${dyIndicesHelper.i2oExpression('dyIndices')}];
                let tmpval = vec4<f32>(xValue * wValue0,
                                      xValue * wValue1,
                                      xValue * wValue2,
                                      xValue * wValue3);
                dotProd[1] = dotProd[1] + tmpval;
              }
            }
          }
        }

        for (var i: u32 = 0; i < ${workPerThread}; i = i + 1) {
          ${
          outputIndicesHelper.indicesVariableDeclaration('outputIndices', [
            'batch', 'r', 'c+i', 'd1'
          ])};
          result[${outputIndicesHelper.i2oExpression('outputIndices')}] = dotProd[i];
        }
      }`;
      const codeSnippet = `
          ${outputIndicesHelper.o2iCall('global_idx', 'outputIndices')}
          let batch = outputIndices[0];
          let d1 = outputIndices[${channelDim}];
          let dyCorner = vec2<i32>(i32(outputIndices[${rowDim}]), i32(outputIndices[${colDim}])) - pads;
          let dyRCorner = dyCorner.x;
          let dyCCorner = dyCorner.y;
          // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
          // ? = to be determined. : = across all values in that axis.
          var dotProd = 0.0;
          for (var wR: u32 = 0; wR < effectiveFilterDims.x; wR = wR + 1) {
            if (wR % dilations.x != 0) {
              continue;
            }
            let dyR = (f32(dyRCorner) + f32(wR)) / f32(strides[0]);
            let wRPerm = filterDims.x - 1 - wR / dilations.x;
            if (dyR < 0.0 || dyR >= f32(outBackprop[1]) || fract(dyR) > 0.0 ||
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
              if (dyC < 0.0 || dyC >= f32(outBackprop[2]) ||
                  fract(dyC) > 0.0 || wCPerm < 0) {
                continue;
              }
              let idyC: u32 = u32(dyC);

              for (var d2: u32 = 0; d2 < outBackprop[3]; d2 = d2 + 1) {
                ${
          dyIndicesHelper.indicesVariableDeclaration(
              'dyIndices',
              isChannelsLast ? ['batch', 'idyR', 'idyC', 'd2'] :
                               [
                                 'batch', 'd2', 'idyR', 'idyC'
                               ])};
                let xValue =  Dy[${dyIndicesHelper.i2oExpression('dyIndices')}];
                  ${
          wIndicesHelper.indicesVariableDeclaration('wIndices', [
            'd2', 'd1', 'wRPerm', 'wCPerm'
          ])};

                let wValue = W[${wIndicesHelper.i2oExpression('wIndices')}];
                dotProd = dotProd + xValue * wValue;
              }
            }
          }
          result[global_idx] = dotProd;
        `;

      return `
${wIndicesHelper.i2oImpl}
  ${dyIndicesHelper.i2oImpl}
  ${outputIndicesHelper.o2iImpl}
  ${declareFunctions}
  ${declareInputs.join('\n')}
  @group(0) @binding(${declareInputs.length}) var<storage, read_write> result: array<${isVec4 ? 'vec4<f32>' : 'f32'}>;
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
    ${outputIndicesHelper.indicesVariableDeclaration('outputIndices')}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)};
  ${isVec4 ? codeSnippet4 : codeSnippet}}`;
    };

export const createConvTranspose2DProgramInfo =
    (inputs: readonly TensorView[], metadata: ProgramMetadata, attributes: ConvTransposeAttributes,
     squeezeOutputShapeFunction?: (shape: readonly number[]) => number[]): ProgramInfo => {
      const hasBias = inputs.length > 2;
      const isChannelsLast = attributes.format === 'NHWC';
      const outputShape = attributes.outputShape;
      const batchSize = outputShape[0];
      const outWidth = outputShape[isChannelsLast ? 1 : 2];
      const outHeight = outputShape[isChannelsLast ? 2 : 3];
      const outChannels = outputShape[isChannelsLast ? 3 : 1];
      const inChannels = inputs[0].dims[isChannelsLast ? 3 : 1];
      const isVec4 = inChannels % 4 === 0 && outChannels % 4 === 0;

      const dispatchX = isChannelsLast ? outChannels : outWidth * outHeight;
      const dispatchY = isChannelsLast ? outWidth * outHeight : outChannels;
      const workGroupSize: [number, number, number] =
          isVec4 ? [8, 8, 1] : [dispatchX <= 4 ? 4 : 16, dispatchX > 4 && dispatchY <= 4 ? 4 : 16, 1];
      const elementsPerThread =
          isVec4 ? [4, 4, 1] : [dispatchX <= 4 ? 1 : 2, dispatchX > 4 && dispatchY <= 4 ? 1 : 2, 1];
      const dispatch = [
        Math.ceil(dispatchX / workGroupSize[0] / elementsPerThread[0]),
        Math.ceil(dispatchY / workGroupSize[1] / elementsPerThread[1]),
        Math.ceil(batchSize / workGroupSize[2] / elementsPerThread[1])
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
            shaderHelper, inputs, attributes, outputShape, hasBias, elementsPerThread),
      };
    };
