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
import {TensorView} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {GpuDataType, ProgramInfo, ProgramMetadata} from '../../types';
import {ConvTransposeAttributes} from '../conv-transpose';

import {Activation, typeSnippet} from './activation_util';
import {utilFunctions} from './conv_util';
import {makeMatMulPackedSource, makeMatMulPackedVec4Source} from './matmul_packed_webgpu';

const conv2dTransposeCommonSnippet =
    (isChannelsLast: boolean, fitAOuter: boolean, fitBOuter: boolean, fitInner: boolean, activation?: Activation,
     innerElementSize = 4): string => {
      const getWSnippet = (innerElementSize: number) => {
        switch (innerElementSize) {
          case 1:
            return 'return W[getIndexFromCoords4D(coord, wShape)];';
          case 4:
            return `
            let coord1 = vec4<i32>(coordX, coordY, col + 1, rowInner);
            let coord2 = vec4<i32>(coordX, coordY, col + 2, rowInner);
            let coord3 = vec4<i32>(coordX, coordY, col + 3, rowInner);
            let v0 = W[getIndexFromCoords4D(coord, wShape)];
            let v1 = W[getIndexFromCoords4D(coord1, wShape)];
            let v2 = W[getIndexFromCoords4D(coord2, wShape)];
            let v3 = W[getIndexFromCoords4D(coord3, wShape)];
            return vec4<f32>(v0, v1, v2, v3);
            `;
          default:
            throw new Error(`innerElementSize ${innerElementSize} is not supported.`);
        }
      };

      const readASnippet = `
      let outWidth = ${isChannelsLast ? 'outShape[2]' : 'outShape[3]'};
      let outRow = row / outWidth;
      let outCol = row % outWidth;

      let WRow = col / (filterDims[1] * outBackprop[3]);
      let WCol = col / outBackprop[3] % filterDims[1];
      let xR = f32(outRow - pad[0] + WRow) / f32(stride[0]);
      let xC = f32(outCol - pad[1] + WCol) / f32(stride[1]);
      if (xR < 0.0 || xR >= f32(outBackprop[1]) || fract(xR) > 0.0) {
        return ${typeSnippet(innerElementSize)}(0.0);
      }
      if (xC < 0.0 || xC >= f32(outBackprop[2]) || fract(xC) > 0.0) {
        return ${typeSnippet(innerElementSize)}(0.0);
      }
      let coord = vec4<i32>(
          batch,
          i32(xR),
          i32(xC),
          col % outBackprop[3]);
      return x[getIndexFromCoords4D(coord, xShape)/${innerElementSize}];`;

      const sampleA = `if (row < dimAOuter && col < dimInner) {
        ${readASnippet}
      }
      return ${typeSnippet(innerElementSize)}(0.0);`;

      const userCode = `
  fn mm_readA(batch: i32, row : i32, col : i32) -> ${typeSnippet(innerElementSize)} {
    ${sampleA}
  }

  fn mm_readB(batch: i32, row : i32, col : i32) -> ${typeSnippet(innerElementSize)} {
    let coordX = filterDims.x - 1 -
        row / (filterDims[1] * outBackprop[3]);
    let coordY = filterDims.y - 1 -
        (row / outBackprop[3]) % filterDims[1];
    if (row < dimInner && col < dimBOuter &&
        coordX >= 0 && coordY >= 0) {
      let rowInner = row % outBackprop[3];
      let coord = vec4<i32>(coordX, coordY, col, rowInner);
      ${getWSnippet(innerElementSize)}
    }
    return ${typeSnippet(innerElementSize)}(0.0);
  }

  fn mm_write(batch: i32, row : i32, col : i32, valueInput : ${typeSnippet(innerElementSize)}) {
    if (row < dimAOuter && col < dimBOuter) {
      var value = valueInput;
      let outWidth =${isChannelsLast ? 'outShape[1]' : 'outShape[2]'};
      let outCoord = vec4<i32>(
          batch,
          row / outWidth,
          row % outWidth,
          col);
      result[getIndexFromCoords4D(outCoord, outShape)/${innerElementSize}] = value;
    }
  }`;
      return userCode;
    };

export const createConv2dTransposeMatMulProgramInfo =
    (inputs: readonly TensorView[], metadata: ProgramMetadata, attributes: ConvTransposeAttributes,
     outputShape: readonly number[], dimAOuter: number, dimBOuter: number, dimInner: number, hasBias: boolean,
     sequentialAccessByThreads: boolean): ProgramInfo => {
      const isChannelsLast = attributes.format === 'NHWC';
      const inChannels = isChannelsLast ? inputs[0].dims[3] : inputs[0].dims[1];
      const batchSize = outputShape[0];
      const outWidth = outputShape[isChannelsLast ? 1 : 2];
      const outHeight = outputShape[isChannelsLast ? 2 : 3];
      const outChannels = outputShape[isChannelsLast ? 3 : 1];
      const isVec4 = (((inChannels % 4 === 0 || inChannels % 3 === 0) && isChannelsLast) ||
                      (outWidth % 4 === 0 && !isChannelsLast)) &&
          outChannels % 4 === 0;

      // TODO: fine tune size
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

      LOG_DEBUG('verbose', () => `[conv2d_backprop_mm_webgpu] dispatch = ${dispatch}`);

      const innerElementSize = isVec4 ? (isChannelsLast && inChannels % 4 !== 0 ? 3 : 4) : elementsPerThread[0];

      const tileAOuter = workGroupSize[1] * elementsPerThread[1];
      const tileBOuter = workGroupSize[0] * elementsPerThread[0];
      const tileInner = Math.max(workGroupSize[0] * innerElementSize, workGroupSize[1]);

      const fitAOuter = dimAOuter % tileAOuter === 0;
      const fitBOuter = dimBOuter % tileBOuter === 0;
      const fitInner = dimInner % tileInner === 0;

      const declareInputs = [
        `@group(0) @binding(0) var<storage, read> x: array<${isVec4 && innerElementSize === 4 ? 'vec4<f32>' : 'f32'}>;`,
        `@group(0) @binding(1) var<storage, read> W: array<${isVec4 ? 'vec4<f32>' : 'f32'}>;`
      ];
      let declareFunctions = `
      fn setOutputAtIndex(flatIndex : i32, value : ${isVec4 ? 'vec4<f32>' : 'f32'}) {
        result[flatIndex] = ${isVec4 ? 'vec4<f32>' : 'f32'}(value);
      }
      fn setOutputAtCoords(d0 : i32, d1 : i32, d2 : i32, d3 : i32, value : ${isVec4 ? 'vec4<f32>' : 'f32'}) {
        let flatIndex = getOutputIndexFromCoords(vec4<i32>(d0, d1, d2, d3));
        setOutputAtIndex(flatIndex ${isVec4 ? '/ 4' : ''}, value);
      }`;
      if (hasBias) {
        declareInputs.push(`@group(0) @binding(2) var<storage, read> bias: array<${isVec4 ? 'vec4<f32>' : 'f32'}>;`);
        declareFunctions += `
        fn getBiasByOutputCoords(coords : vec4<i32>) -> ${isVec4 ? 'vec4<f32>' : 'f32'} {
          return bias[coords.${isChannelsLast ? 'w' : 'y'}${isVec4 ? '/ 4' : ''}];
        }`;
      }

      return {
        ...metadata,
        outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
        dispatchGroup: () => ({x: dispatch[0], y: dispatch[1], z: dispatch[2]}),
        getShaderSource: () => `
        ${utilFunctions}
        //struct Uniforms { xShape : vec4<i32>, wShape : vec4<i32>, outShape : vec4<i32>,
        //  outShapeStrides: vec3<i32>, filterDims : vec2<i32>, pad : vec2<i32>, stride : vec2<i32>,
        //  dilation : vec2<i32>, dimAOuter : i32, dimBOuter : i32, dimInner : i32 };
        ${declareInputs.join('')}
        @group(0) @binding(${declareInputs.length}) var<storage, read_write> result: array<${
            isVec4 ? 'vec4<f32>' : 'f32'}>;
        //@group(0) @binding(${declareInputs.length + 1}) var<uniform> uniforms: Uniforms;

        const xShape : vec4<i32> = vec4<i32>(${inputs[0].dims.join(',')});
        const wShape : vec4<i32> = vec4<i32>(${inputs[1].dims.join(',')});
        const outShape : vec4<i32> = vec4<i32>(${outputShape.join(',')});
        const outShapeStrides : vec3<i32> = vec3<i32>(${ShapeUtil.computeStrides(outputShape).slice(0, 3).join(',')});
        const filterDims : vec2<i32> = vec2<i32>(${attributes.kernelShape[isChannelsLast ? 1 : 2]}, ${
            attributes.kernelShape[isChannelsLast ? 2 : 3]});
        const pad : vec2<i32> = vec2<i32>(${attributes.pads[0]}, ${attributes.pads[1]});
        const stride : vec2<i32> = vec2<i32>(${attributes.strides[0]}, ${attributes.strides[1]});
        const dilation : vec2<i32> = vec2<i32>(${attributes.dilations[0]}, ${attributes.dilations[1]});
        const outBackprop : vec4<i32> = outShape;
        const dimAOuter : i32 = ${dimAOuter};
        const dimBOuter : i32 = ${dimBOuter};
        const dimInner : i32 = ${dimInner};
        ${declareFunctions}
        ${conv2dTransposeCommonSnippet(isChannelsLast, fitAOuter, fitBOuter, fitInner, undefined, isVec4 ? 4 : 1)}
            ${
            isVec4 ? makeMatMulPackedVec4Source(elementsPerThread, workGroupSize, !isChannelsLast, tileInner) :
                     makeMatMulPackedSource(
                         elementsPerThread, workGroupSize, !isChannelsLast, tileInner, false, undefined,
                         sequentialAccessByThreads)}`
      };
    };
