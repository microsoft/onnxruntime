/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

// sampled from [@tensorflow/tfjs] tfjs-backend-webgpu/src/conv2d_mm_webgpu.ts
//
// modified to fit the needs of the project

import {LOG_DEBUG} from '../../../log';
import {TensorView} from '../../../tensor-view';
import {ProgramInfo, ProgramUniform} from '../../types';
import {createTensorShapeVariables, inputVariable, outputVariable, ShaderHelper, tensorTypeToWsglStorageType} from '../common';
import {ConvAttributes} from '../conv';
import {getActivationSnippet} from '../fuse-utils';

import {biasSnippet, typeSnippet} from './activation_util';
import {utilFunctions} from './conv_util';
import {makeMatMulPackedSource, makeMatMulPackedVec4Source} from './matmul_packed_webgpu';

const conv2dCommonSnippet =
    (isChannelsLast: boolean, fitAOuter: boolean, fitBOuter: boolean, fitInner: boolean, addBias = false,
     attributes: ConvAttributes, innerElementSizeX = 4, innerElementSizeW = 4, innerElementSize = 4,
     dataType = 'f32'): string => {
      const getXSnippet = (innerElementSize: number) => {
        switch (innerElementSize) {
          case 1:
            return 'resData = x[xIndex];';
          case 3:
            return `resData = vec3<${dataType}>(x[xIndex], x[xIndex + 1], x[xIndex + 2]);`;
          case 4:
            return 'resData = x[xIndex / 4];';
          default:
            throw new Error(`innerElementSize ${innerElementSize} is not supported.`);
        }
      };
      const getWSnippet = (innerElementSize: number) => {
        switch (innerElementSize) {
          case 1:
            return 'return w[row * i32(uniforms.w_shape[3]) + colIn];';
          case 4:
            return 'return w[row * i32(uniforms.w_shape[3]) / 4 + colIn];';
          default:
            throw new Error(`innerElementSize ${innerElementSize} is not supported.`);
        }
      };
      const coordASnippet = isChannelsLast ? `
    let coord = vec4<i32>(batch, xRow, xCol, xCh);
    ` :
                                             `
    let coord = vec4<i32>(batch, xCh, xRow, xCol);
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

      const xHeight = isChannelsLast ? 'i32(uniforms.x_shape[1])' : 'i32(uniforms.x_shape[2])';
      const xWidth = isChannelsLast ? 'i32(uniforms.x_shape[2])' : 'i32(uniforms.x_shape[3])';
      const row = isChannelsLast ? 'row' : 'col';
      const col = isChannelsLast ? 'col' : 'row';
      const readXSnippet = `
    let inChannels = i32(uniforms.w_shape[2]);
    let outWidth = ${isChannelsLast ? 'i32(uniforms.result_shape[2])' : 'i32(uniforms.result_shape[3])'};
    let outRow = ${row} / outWidth;
    let outCol = ${row} % outWidth;

    let WRow = ${col} / (filterDims[1] * inChannels);
    let WCol = ${col} / inChannels % filterDims[1];
    let xRow = outRow * stride[0] + dilation[0] * WRow - pad[0];
    let xCol = outCol * stride[1] + dilation[1] * WCol - pad[1];
    let xCh = ${col} % inChannels;
    var resData = ${typeSnippet(innerElementSizeX, dataType)}(0.0);
    // The bounds checking is always needed since we use it to pad zero for
    // the 'same' padding type.
    if (xRow >= 0 && xRow < ${xHeight} && xCol >= 0 && xCol < ${xWidth}) {
      ${coordASnippet}
      let xIndex = getIndexFromCoords4D(coord, vec4<i32>(uniforms.x_shape));
      ${getXSnippet(innerElementSizeX)}
    }
    return resData;`;

      const sampleX = isChannelsLast ? (fitAOuter && fitInner ? `
    let col = colIn * ${innerElementSizeX};
    ${readXSnippet}` :
                                                                `
    let col = colIn * ${innerElementSizeX};
    if (row < uniforms.dimAOuter && col < uniforms.dimInner) {
      ${readXSnippet}
    }
    return ${typeSnippet(innerElementSizeX, dataType)}(0.0);`) :
                                       (fitInner && fitBOuter ? `
    let col = colIn * ${innerElementSizeX};
    ${readXSnippet}` :
                                                                `
    let col = colIn * ${innerElementSizeX};
    if (row < uniforms.dimInner && col < uniforms.dimBOuter) {
      ${readXSnippet}
    }
    return ${typeSnippet(innerElementSizeX, dataType)}(0.0);`);

      const sampleW = `${getWSnippet(innerElementSizeW)}`;

      const resType = typeSnippet(innerElementSize, dataType);
      const aType =
          isChannelsLast ? typeSnippet(innerElementSizeX, dataType) : typeSnippet(innerElementSizeW, dataType);
      const bType =
          isChannelsLast ? typeSnippet(innerElementSizeW, dataType) : typeSnippet(innerElementSizeX, dataType);
      const {activationFunction, applyActivation} = getActivationSnippet(attributes, resType);
      const userCode = `
    ${activationFunction}
    fn mm_readA(batch: i32, row : i32, colIn : i32) -> ${aType} {
      ${isChannelsLast ? sampleX : sampleW}
    }

    fn mm_readB(batch: i32, row : i32, colIn : i32) -> ${bType} {
      ${isChannelsLast ? sampleW : sampleX}
    }

    fn mm_write(batch: i32, row : i32, colIn : i32, valueIn : ${resType}) {
      let col = colIn * ${innerElementSize};
      if (row < uniforms.dimAOuter && col < uniforms.dimBOuter)
      {
      var value = valueIn;
      let outWidth = ${isChannelsLast ? 'i32(uniforms.result_shape[2])' : 'i32(uniforms.result_shape[3])'};
      ${coordResSnippet}
      ${biasSnippet(addBias)}
      ${applyActivation}
      setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
      }
    }`;
      return userCode;
    };

export const createConv2DMatMulProgramInfo =
    (inputs: readonly TensorView[], attributes: ConvAttributes, outputShape: readonly number[], dimAOuter: number,
     dimBOuter: number, dimInner: number, hasBias: boolean, sequentialAccessByThreads: boolean): ProgramInfo => {
      const isChannelsLast = attributes.format === 'NHWC';
      const inChannels = isChannelsLast ? inputs[0].dims[3] : inputs[0].dims[1];
      const batchSize = outputShape[0];
      const outWidth = isChannelsLast ? outputShape[2] : outputShape[3];
      const outHeight = isChannelsLast ? outputShape[1] : outputShape[2];
      const outChannels = isChannelsLast ? outputShape[3] : outputShape[1];
      // TODO: enable vec4 for NCHW
      const isVec4 = isChannelsLast && (inChannels % 4 === 0 || inChannels % 3 === 0) && outChannels % 4 === 0;

      // TODO: fine tune size
      const dispatchX = isChannelsLast ? outChannels : outWidth * outHeight;
      const dispatchY = isChannelsLast ? outWidth * outHeight : outChannels;
      const workGroupSize: [number, number, number] = [8, 8, 1];
      const elementsPerThread = dimAOuter <= 8 ? [4, 1, 1] : [4, 4, 1];
      const dispatch = [
        Math.ceil(dispatchX / workGroupSize[0] / elementsPerThread[0]),
        Math.ceil(dispatchY / workGroupSize[1] / elementsPerThread[1]),
        Math.ceil(batchSize / workGroupSize[2] / elementsPerThread[2])
      ];

      LOG_DEBUG('verbose', () => `[conv2d_mm_webgpu] dispatch = ${dispatch}`);

      const innerElementSize = isVec4 ? (isChannelsLast && inChannels % 4 !== 0 ? 3 : 4) : 1;

      const tileAOuter = workGroupSize[1] * elementsPerThread[1];
      const tileBOuter = workGroupSize[0] * elementsPerThread[0];
      const tileInner = Math.max(workGroupSize[0] * innerElementSize, workGroupSize[1]);

      const fitAOuter = dimAOuter % tileAOuter === 0;
      const fitBOuter = dimBOuter % tileBOuter === 0;
      const fitInner = dimInner % tileInner === 0;

      const elementsSize = isVec4 ? [innerElementSize, 4, 4] : [1, 1, 1];
      const t = tensorTypeToWsglStorageType(inputs[0].dataType);

      // TODO: support component 2, 3.
      const components = isVec4 ? 4 : 1;
      const programUniforms: ProgramUniform[] =
          [{type: 'int32', data: dimAOuter}, {type: 'int32', data: dimBOuter}, {type: 'int32', data: dimInner}];
      const x =
          inputVariable('x', inputs[0].dataType, inputs[0].dims.length, innerElementSize === 3 ? 1 : innerElementSize);
      const w = inputVariable('w', inputs[1].dataType, inputs[1].dims.length, components);
      const inputVariables = [x, w];

      programUniforms.push(...createTensorShapeVariables(inputs[0].dims));
      programUniforms.push(...createTensorShapeVariables(inputs[1].dims));

      let declareFunctions = `
      fn setOutputAtIndex(flatIndex : i32, value : ${isVec4 ? `vec4<${t}>` : t}) {
        result[flatIndex] = ${isVec4 ? `vec4<${t}>` : t}(value);
      }
      fn setOutputAtCoords(d0 : i32, d1 : i32, d2 : i32, d3 : i32, value : ${isVec4 ? `vec4<${t}>` : t}) {
        let flatIndex = getOutputIndexFromCoords(vec4<i32>(d0, d1, d2, d3));
        setOutputAtIndex(flatIndex ${isVec4 ? '/ 4' : ''}, value);
      }`;
      if (hasBias) {
        const bias = inputVariable('bias', inputs[2].dataType, inputs[2].dims.length, components);
        inputVariables.push(bias);

        programUniforms.push(...createTensorShapeVariables(inputs[2].dims));

        declareFunctions += `
        fn getBiasByOutputCoords(coords : vec4<i32>) -> ${isVec4 ? `vec4<${t}>` : t} {
          return bias[coords.${isChannelsLast ? 'w' : 'y'}${isVec4 ? '/ 4' : ''}];
        }`;
      }
      const output = outputVariable('result', inputs[0].dataType, outputShape.length, components);
      programUniforms.push(...createTensorShapeVariables(outputShape));
      return {
        name: 'Conv2DMatMul',
        shaderCache: {hint: attributes.cacheKey},
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
          dispatchGroup: {x: dispatch[0], y: dispatch[1], z: dispatch[2]},
          programUniforms,
        }),
        getShaderSource: (shaderHelper: ShaderHelper) => `
        ${utilFunctions('uniforms.result_strides')}
        //struct Uniforms { xShape : vec4<i32>, wShape : vec4<i32>, outShape : vec4<i32>,
        //  outShapeStrides: vec3<i32>, filterDims : vec2<i32>, pad : vec2<i32>, stride : vec2<i32>,
        //  dilation : vec2<i32>, dimAOuter : i32, dimBOuter : i32, dimInner : i32 };
        ${
            shaderHelper.registerUniform('dimAOuter', 'i32')
                .registerUniform('dimBOuter', 'i32')
                .registerUniform('dimInner', 'i32')
                .declareVariables(...inputVariables, output)}
        const filterDims : vec2<i32> = vec2<i32>(${attributes.kernelShape[0]}, ${attributes.kernelShape[1]});
        const pad : vec2<i32> = vec2<i32>(${attributes.pads[0]}, ${attributes.pads[1]});
        const stride : vec2<i32> = vec2<i32>(${attributes.strides[0]}, ${attributes.strides[1]});
        const dilation : vec2<i32> = vec2<i32>(${attributes.dilations[0]}, ${attributes.dilations[1]});
        ${declareFunctions}
        ${
            conv2dCommonSnippet(
                isChannelsLast, fitAOuter, fitBOuter, fitInner, hasBias, attributes, elementsSize[0], elementsSize[1],
                elementsSize[2], t)}
            ${
            isVec4 ?
                makeMatMulPackedVec4Source(elementsPerThread, workGroupSize, t, undefined, !isChannelsLast, tileInner) :
                makeMatMulPackedSource(
                    elementsPerThread, workGroupSize, t, undefined, !isChannelsLast, tileInner, false, undefined,
                    sequentialAccessByThreads)}`
      };
    };
