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

// sampled from [@tensorflow/tfjs] tfjs-backend-webgpu/src/conv_backprop_mm_webgpu.ts
//
// modified to fit the needs of the project

import {LOG_DEBUG} from '../../../log';
import {TensorView} from '../../../tensor-view';
import {ProgramInfo, ProgramUniform} from '../../types';
import {createTensorShapeVariables, enableShapesUniforms, inputVariable, outputVariable, ShaderHelper} from '../common';
import {ConvTransposeAttributes} from '../conv-transpose';
import {getActivationSnippet} from '../fuse-utils';

import {biasSnippet, typeSnippet} from './activation_util';
import {utilFunctions} from './conv_util';
import {makeMatMulPackedSource, makeMatMulPackedVec4Source} from './matmul_packed_webgpu';

const conv2dTransposeCommonSnippet =
    (xShapeStr: string, wShapeStr: string, outputShapeStr: string, isChannelsLast: boolean, addBias = false,
     attributes: ConvTransposeAttributes, innerElementSize = 4): string => {
      const type = typeSnippet(innerElementSize, 'f32');
      const getWSnippet = (innerElementSize: number) => {
        switch (innerElementSize) {
          case 1:
            return `return w[getIndexFromCoords4D(coord, vec4<i32>(${wShapeStr}))];`;
          case 4:
            return `
            let coord1 = vec4<i32>(coordX, coordY, col + 1, rowInner);
            let coord2 = vec4<i32>(coordX, coordY, col + 2, rowInner);
            let coord3 = vec4<i32>(coordX, coordY, col + 3, rowInner);
            let v0 = w[getIndexFromCoords4D(coord, vec4<i32>(${wShapeStr}))];
            let v1 = w[getIndexFromCoords4D(coord1, vec4<i32>(${wShapeStr}))];
            let v2 = w[getIndexFromCoords4D(coord2, vec4<i32>(${wShapeStr}))];
            let v3 = w[getIndexFromCoords4D(coord3, vec4<i32>(${wShapeStr}))];
            return vec4<f32>(v0, v1, v2, v3);
            `;
          default:
            throw new Error(`innerElementSize ${innerElementSize} is not supported.`);
        }
      };
      const coordASnippet = isChannelsLast ? `
      let coord = vec4<i32>(batch, iXR, iXC, xCh);
      ` :
                                             `
      let coord = vec4<i32>(batch, xCh, iXR, iXC);
      `;

      const coordResSnippet = isChannelsLast ? `
    let coords = vec4<i32>(
      batch,
      row / outWidth,
      row % outWidth,
      col);
    ` :
                                               `
    let coords = vec4<i32>(
      batch,
      row,
      col / outWidth,
      col % outWidth);
    `;

      const xHeight = isChannelsLast ? 'outBackprop[1]' : 'outBackprop[2]';
      const xWidth = isChannelsLast ? 'outBackprop[2]' : 'outBackprop[3]';
      const row = isChannelsLast ? 'row' : 'col';
      const col = isChannelsLast ? 'col' : 'row';

      const readASnippet = `
      let inChannels = ${isChannelsLast ? 'outBackprop[3]' : 'outBackprop[1]'};
      let outWidth = ${isChannelsLast ? `i32(${outputShapeStr}[2])` : `i32(${outputShapeStr}[3])`};
      let outRow = ${row} / outWidth;
      let outCol = ${row} % outWidth;

      let WRow = ${col} / (filterDims[1] * inChannels);
      let WCol = ${col} / inChannels % filterDims[1];
      let xR = f32(outRow - pads[0] + dilation[0] * WRow) / f32(strides[0]);
      let xC = f32(outCol - pads[1] + dilation[1] * WCol) / f32(strides[1]);
      if (xR < 0.0 || xR >= f32(${xHeight}) || fract(xR) > 0.0) {
        return ${type}(0.0);
      }
      if (xC < 0.0 || xC >= f32(${xWidth}) || fract(xC) > 0.0) {
        return ${type}(0.0);
      }
      let iXR = i32(xR);
      let iXC = i32(xC);
      let xCh = ${col} % inChannels;
      ${coordASnippet}
      return x[getIndexFromCoords4D(coord, vec4<i32>(${xShapeStr}))/${innerElementSize}];`;

      const sampleA = isChannelsLast ? `
      let col = colIn * ${innerElementSize};
      if (row < uniforms.dimAOuter && col < uniforms.dimInner) {
        ${readASnippet}
      }
      return ${type}(0.0);` :
                                       `
      let col = colIn * ${innerElementSize};
      if (row < uniforms.dimInner && col < uniforms.dimBOuter) {
        ${readASnippet}
      }
      return ${type}(0.0);`;

      const sampleW = `
      let col = colIn * ${innerElementSize};
      let inChannels = ${isChannelsLast ? 'outBackprop[3]' : 'outBackprop[1]'};
      let coordX = filterDims.x - 1 - row / (filterDims[1] * inChannels);
      let coordY = filterDims.y - 1 - (row / inChannels) % filterDims[1];
      if (${
          isChannelsLast ? 'row < uniforms.dimInner && col < uniforms.dimBOuter' :
                           'row < uniforms.dimInner && col < uniforms.dimAOuter'}  && coordX >= 0 && coordY >= 0) {
        let rowInner = row % inChannels;
        let coord = vec4<i32>(coordX, coordY, col, rowInner);
        ${getWSnippet(innerElementSize)}
      }
      return ${type}(0.0);
      `;

      const {activationFunction, applyActivation} = getActivationSnippet(attributes, type);
      const userCode = `
      ${activationFunction}
  fn mm_readA(batch: i32, row : i32, colIn : i32) -> ${type} {
    ${isChannelsLast ? sampleA : sampleW}
  }

  fn mm_readB(batch: i32, row : i32, colIn : i32) -> ${type} {
    ${isChannelsLast ? sampleW : sampleA}
  }

  fn mm_write(batch: i32, row : i32, colIn : i32, valueInput : ${type}) {
    let col = colIn * ${innerElementSize};
    if (row < uniforms.dimAOuter && col < uniforms.dimBOuter) {
      var value = valueInput;
      let outWidth = ${isChannelsLast ? `i32(${outputShapeStr}[2])` : `i32(${outputShapeStr}[3])`};
      ${coordResSnippet}
      ${biasSnippet(addBias)}
      ${applyActivation}
      result[getIndexFromCoords4D(coords, vec4<i32>(${outputShapeStr}))/${innerElementSize}] = value;
    }
  }`;
      return userCode;
    };

export const createConv2DTransposeMatMulProgramInfo =
    (inputs: readonly TensorView[], attributes: ConvTransposeAttributes, outputShape: readonly number[],
     dimAOuter: number, dimBOuter: number, dimInner: number, hasBias: boolean,
     sequentialAccessByThreads: boolean): ProgramInfo => {
      const isChannelsLast = attributes.format === 'NHWC';
      const inChannels = isChannelsLast ? inputs[0].dims[3] : inputs[0].dims[1];
      const batchSize = outputShape[0];
      const outWidth = isChannelsLast ? outputShape[2] : outputShape[3];
      const outHeight = isChannelsLast ? outputShape[1] : outputShape[2];
      const outChannels = isChannelsLast ? outputShape[3] : outputShape[1];
      const isVec4 =
          isChannelsLast ? inChannels % 4 === 0 && outChannels % 4 === 0 : outWidth % 4 === 0 && outChannels % 4 === 0;

      // TODO: fine tune size
      const dispatchX = isChannelsLast ? outChannels : outWidth * outHeight;
      const dispatchY = isChannelsLast ? outWidth * outHeight : outChannels;
      const workGroupSize: [number, number, number] = isVec4 ?
          [8, 8, 1] :
          [(dispatchX <= 4 || dispatchY <= 4) ? 4 : 16, dispatchX > 4 && dispatchY <= 4 ? 4 : 16, 1];
      const elementsPerThread =
          isVec4 ? [4, 4, 1] : [dispatchX <= 4 ? 1 : 4, dispatchX > 4 && dispatchY <= 4 ? 1 : 4, 1];
      const dispatch = [
        Math.ceil(dispatchX / workGroupSize[0] / elementsPerThread[0]),
        Math.ceil(dispatchY / workGroupSize[1] / elementsPerThread[1]),
        Math.ceil(batchSize / workGroupSize[2] / elementsPerThread[2])
      ];

      LOG_DEBUG('verbose', () => `[conv_backprop_mm_webgpu] dispatch = ${dispatch}`);

      const innerElementSize = isVec4 ? 4 : 1;
      const tileInner = Math.max(workGroupSize[0] * innerElementSize, workGroupSize[1]);
      const components = isVec4 ? 4 : 1;
      const programUniforms: ProgramUniform[] =
          [{type: 'int32', data: dimAOuter}, {type: 'int32', data: dimBOuter}, {type: 'int32', data: dimInner}];
      const enableXShapesUniforms = enableShapesUniforms(inputs[0].dims.length);
      const xShapeOrRank = enableXShapesUniforms ? inputs[0].dims.length : inputs[0].dims;
      const enableWShapesUniforms = enableShapesUniforms(inputs[1].dims.length);
      const wShapeOrRank = enableWShapesUniforms ? inputs[1].dims.length : inputs[1].dims;
      const x = inputVariable('x', inputs[0].dataType, xShapeOrRank, components);
      const w = inputVariable('w', inputs[1].dataType, wShapeOrRank, 1);
      const enableOutputShapesUniforms = enableShapesUniforms(outputShape.length);
      const outputShapeOrRank = enableOutputShapesUniforms ? outputShape.length : outputShape;
      const output = outputVariable('result', inputs[0].dataType, outputShapeOrRank, components);
      const inputVariables = [x, w];
      if (enableXShapesUniforms) {
        programUniforms.push(...createTensorShapeVariables(inputs[0].dims));
      }
      if (enableWShapesUniforms) {
        programUniforms.push(...createTensorShapeVariables(inputs[1].dims));
      }

      let declareFunctions = '';
      if (hasBias) {
        const enableBiasShapesUniforms = enableShapesUniforms(inputs[2].dims.length);
        const biasShapeOrRank = enableBiasShapesUniforms ? inputs[2].dims.length : inputs[2].dims;
        const bias = inputVariable('bias', inputs[2].dataType, biasShapeOrRank, components);
        inputVariables.push(bias);
        if (enableBiasShapesUniforms) {
          programUniforms.push(...createTensorShapeVariables(inputs[2].dims));
        }
        declareFunctions += `
        fn getBiasByOutputCoords(coords : vec4<i32>) -> ${isVec4 ? 'vec4<f32>' : 'f32'} {
          return bias[coords.${isChannelsLast ? 'w' : 'y'}${isVec4 ? '/ 4' : ''}];
        }`;
      }

      if (enableOutputShapesUniforms) {
        programUniforms.push(...createTensorShapeVariables(outputShape));
      }

      const xShapeStr = enableXShapesUniforms ? 'uniforms.x_shape' : 'x_shape';
      const wShapeStr = enableWShapesUniforms ? 'uniforms.w_shape' : 'w_shape';
      const outputShapeStr = enableOutputShapesUniforms ? 'uniforms.result_shape' : 'result_shape';

      return {
        name: 'Conv2DTransposeMatMul',
        shaderCache: {hint: attributes.cacheKey},
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
          dispatchGroup: {x: dispatch[0], y: dispatch[1], z: dispatch[2]},
          programUniforms
        }),
        getShaderSource: (shaderHelper: ShaderHelper) => `
        ${utilFunctions(enableOutputShapesUniforms ? 'uniforms.result_strides' : 'result_strides')}
        ${
            shaderHelper.registerUniform('dimAOuter', 'i32')
                .registerUniform('dimBOuter', 'i32')
                .registerUniform('dimInner', 'i32')
                .declareVariables(...inputVariables, output)};
        const outBackprop : vec4<i32> = vec4<i32>(${inputs[0].dims.join(',')});
        const filterDims : vec2<i32> = vec2<i32>(${attributes.kernelShape[isChannelsLast ? 1 : 2]}, ${
            attributes.kernelShape[isChannelsLast ? 2 : 3]});
        const effectiveFilterDims : vec2<i32> = filterDims + vec2<i32>(
              ${
            attributes.dilations[0] <= 1 ?
                0 :
                (attributes.kernelShape[isChannelsLast ? 1 : 2] - 1) * (attributes.dilations[0] - 1)},
              ${
            attributes.dilations[1] <= 1 ?
                0 :
                (attributes.kernelShape[isChannelsLast ? 2 : 3] - 1) * (attributes.dilations[1] - 1)});
        const pads : vec2<i32> = vec2<i32>(i32(effectiveFilterDims[0]) - 1 - (${
            attributes.pads[0] + attributes.pads[2]})/2,
                                         i32(effectiveFilterDims[1]) - 1 - (${
            attributes.pads[1] + attributes.pads[3]})/2);
        const strides : vec2<i32> = vec2<i32>(${attributes.strides[0]}, ${attributes.strides[1]});
        const dilation : vec2<i32> = vec2<i32>(${attributes.dilations[0]}, ${attributes.dilations[1]});
        const dimAOuter : i32 = ${dimAOuter};
        const dimBOuter : i32 = ${dimBOuter};
        const dimInner : i32 = ${dimInner};
        ${declareFunctions}
        ${
            conv2dTransposeCommonSnippet(
                xShapeStr, wShapeStr, outputShapeStr, isChannelsLast, hasBias, attributes, innerElementSize)}
        ${
            isVec4 ? makeMatMulPackedVec4Source(
                         elementsPerThread, workGroupSize, 'f32', undefined, !isChannelsLast, tileInner) :
                     makeMatMulPackedSource(
                         elementsPerThread, workGroupSize, 'f32', undefined, !isChannelsLast, tileInner, false,
                         undefined, sequentialAccessByThreads)}`
      };
    };
