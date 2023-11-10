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
import {ShapeUtil} from '../../../util';
import {ProgramInfo} from '../../types';
import {tensorTypeToWsglStorageType} from '../common';
import {ConvAttributes} from '../conv';

import {Activation, activationFnSnippet, biasActivationSnippet, typeSnippet} from './activation_util';
import {utilFunctions} from './conv_util';
import {makeMatMulPackedSource, makeMatMulPackedVec4Source} from './matmul_packed_webgpu';

const conv2dCommonSnippet =
    (isChannelsLast: boolean, fitAOuter: boolean, fitBOuter: boolean, fitInner: boolean, addBias = false,
     activation?: Activation, hasPreluActivationWeights = false, innerElementSizeX = 4, innerElementSizeW = 4,
     innerElementSize = 4, dataType = 'f32'): string => {
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
            return 'return w[row * wShape[3] + colIn];';
          case 4:
            return 'return w[row * wShape[3] / 4 + colIn];';
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

      const xHeight = isChannelsLast ? 'xShape[1]' : 'xShape[2]';
      const xWidth = isChannelsLast ? 'xShape[2]' : 'xShape[3]';
      const row = isChannelsLast ? 'row' : 'col';
      const col = isChannelsLast ? 'col' : 'row';
      const readXSnippet = `
    let inChannels = wShape[2];
    let outWidth = ${isChannelsLast ? 'outShape[2]' : 'outShape[3]'};
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
      let xIndex = getIndexFromCoords4D(coord, xShape);
      ${getXSnippet(innerElementSizeX)}
    }
    return resData;`;

      const sampleX = isChannelsLast ? (fitAOuter && fitInner ? `
    let col = colIn * ${innerElementSizeX};
    ${readXSnippet}` :
                                                                `
    let col = colIn * ${innerElementSizeX};
    if (row < dimAOuter && col < dimInner) {
      ${readXSnippet}
    }
    return ${typeSnippet(innerElementSizeX, dataType)}(0.0);`) :
                                       (fitInner && fitBOuter ? `
    let col = colIn * ${innerElementSizeX};
    ${readXSnippet}` :
                                                                `
    let col = colIn * ${innerElementSizeX};
    if (row < dimInner && col < dimBOuter) {
      ${readXSnippet}
    }
    return ${typeSnippet(innerElementSizeX, dataType)}(0.0);`);

      const sampleW = `${getWSnippet(innerElementSizeW)}`;

      const resType = typeSnippet(innerElementSize, dataType);
      const aType =
          isChannelsLast ? typeSnippet(innerElementSizeX, dataType) : typeSnippet(innerElementSizeW, dataType);
      const bType =
          isChannelsLast ? typeSnippet(innerElementSizeW, dataType) : typeSnippet(innerElementSizeX, dataType);
      const userCode = `
    ${activationFnSnippet(activation, hasPreluActivationWeights, innerElementSize === 4, 4)}
    fn mm_readA(batch: i32, row : i32, colIn : i32) -> ${aType} {
      ${isChannelsLast ? sampleX : sampleW}
    }

    fn mm_readB(batch: i32, row : i32, colIn : i32) -> ${bType} {
      ${isChannelsLast ? sampleW : sampleX}
    }

    fn mm_write(batch: i32, row : i32, colIn : i32, valueIn : ${resType}) {
      let col = colIn * ${innerElementSize};
      if (row < dimAOuter && col < dimBOuter)
      {
      var value = valueIn;
      let outWidth = ${isChannelsLast ? 'outShape[2]' : 'outShape[3]'};
      ${coordResSnippet}
      ${biasActivationSnippet(addBias, activation)}
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

      const innerElementSize = isVec4 ? (isChannelsLast && inChannels % 4 !== 0 ? 3 : 4) : elementsPerThread[0];

      const tileAOuter = workGroupSize[1] * elementsPerThread[1];
      const tileBOuter = workGroupSize[0] * elementsPerThread[0];
      const tileInner = Math.max(workGroupSize[0] * innerElementSize, workGroupSize[1]);

      const fitAOuter = dimAOuter % tileAOuter === 0;
      const fitBOuter = dimBOuter % tileBOuter === 0;
      const fitInner = dimInner % tileInner === 0;

      const elementsSize = isVec4 ? [innerElementSize, 4, 4] : [1, 1, 1];
      const t = tensorTypeToWsglStorageType(inputs[0].dataType);

      const declareInputs = [
        `@group(0) @binding(0) var<storage, read> x: array<${isVec4 && innerElementSize === 4 ? `vec4<${t}>` : t}>;`,
        `@group(0) @binding(1) var<storage, read> w: array<${isVec4 ? `vec4<${t}>` : t}>;`
      ];
      let declareFunctions = `
      fn setOutputAtIndex(flatIndex : i32, value : ${isVec4 ? `vec4<${t}>` : t}) {
        result[flatIndex] = ${isVec4 ? `vec4<${t}>` : t}(value);
      }
      fn setOutputAtCoords(d0 : i32, d1 : i32, d2 : i32, d3 : i32, value : ${isVec4 ? `vec4<${t}>` : t}) {
        let flatIndex = getOutputIndexFromCoords(vec4<i32>(d0, d1, d2, d3));
        setOutputAtIndex(flatIndex ${isVec4 ? '/ 4' : ''}, value);
      }`;
      if (hasBias) {
        declareInputs.push(`@group(0) @binding(2) var<storage, read> bias: array<${isVec4 ? `vec4<${t}>` : t}>;`);
        declareFunctions += `
        fn getBiasByOutputCoords(coords : vec4<i32>) -> ${isVec4 ? `vec4<${t}>` : t} {
          return bias[coords.${isChannelsLast ? 'w' : 'y'}${isVec4 ? '/ 4' : ''}];
        }`;
      }

      return {
        name: 'Conv2DMatMul',
        shaderCache: {hint: attributes.cacheKey},
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
          dispatchGroup: {x: dispatch[0], y: dispatch[1], z: dispatch[2]},
        }),
        getShaderSource: () => `
        ${utilFunctions}
        //struct Uniforms { xShape : vec4<i32>, wShape : vec4<i32>, outShape : vec4<i32>,
        //  outShapeStrides: vec3<i32>, filterDims : vec2<i32>, pad : vec2<i32>, stride : vec2<i32>,
        //  dilation : vec2<i32>, dimAOuter : i32, dimBOuter : i32, dimInner : i32 };
        ${declareInputs.join('')}
        @group(0) @binding(${declareInputs.length}) var<storage, read_write> result: array<${
            isVec4 ? `vec4<${t}>` : t}>;
        //@group(0) @binding(${declareInputs.length + 1}) var<uniform> uniforms: Uniforms;

        const xShape : vec4<i32> = vec4<i32>(${inputs[0].dims.join(',')});
        const wShape : vec4<i32> = vec4<i32>(${inputs[1].dims.join(',')});
        const outShape : vec4<i32> = vec4<i32>(${outputShape.join(',')});
        const outShapeStrides : vec3<i32> = vec3<i32>(${ShapeUtil.computeStrides(outputShape).slice(0, 3).join(',')});
        const filterDims : vec2<i32> = vec2<i32>(${attributes.kernelShape[0]}, ${attributes.kernelShape[1]});
        const pad : vec2<i32> = vec2<i32>(${attributes.pads[0]}, ${attributes.pads[1]});
        const stride : vec2<i32> = vec2<i32>(${attributes.strides[0]}, ${attributes.strides[1]});
        const dilation : vec2<i32> = vec2<i32>(${attributes.dilations[0]}, ${attributes.dilations[1]});
        const dimAOuter : i32 = ${dimAOuter};
        const dimBOuter : i32 = ${dimBOuter};
        const dimInner : i32 = ${dimInner};
        ${declareFunctions}
        ${
            conv2dCommonSnippet(
                isChannelsLast, fitAOuter, fitBOuter, fitInner, hasBias,
                attributes.activation.toLowerCase() as Activation, false, elementsSize[0], elementsSize[1],
                elementsSize[2], t)}
            ${
            isVec4 ?
                makeMatMulPackedVec4Source(elementsPerThread, workGroupSize, t, undefined, !isChannelsLast, tileInner) :
                makeMatMulPackedSource(
                    elementsPerThread, workGroupSize, t, undefined, !isChannelsLast, tileInner, false, undefined,
                    sequentialAccessByThreads)}`
      };
    };
