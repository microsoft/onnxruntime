/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

// sampled from [@tensorflow/tfjs] tfjs-core/src/ops/conv_util.ts
//
// modified to fit the needs of the project

export const utilFunctions = `
fn getIndexFromCoords4D(coords : vec4<i32>, shape : vec4<i32>) -> i32 {
  return dot(coords, vec4<i32>(
      shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1));
}
fn getOutputIndexFromCoords(coords : vec4<i32>) -> i32 {
  return dot(coords, vec4<i32>(
    outShapeStrides.x, outShapeStrides.y, outShapeStrides.z, 1));
}
`;
// type PadType = 'SAME'|'VALID'|'NUMBER'|'EXPLICIT';

// export interface PadInfo {
//   top: number;
//   left: number;
//   right: number;
//   bottom: number;
//   type: PadType;
// }

// /**
//  * Information about the forward pass of a convolution/pooling operation.
//  * It includes input and output shape, strides, filter size and padding
//  * information.
//  */
// export interface Conv2DInfo {
//   batchSize: number;
//   inHeight: number;
//   inWidth: number;
//   inChannels: number;
//   outHeight: number;
//   outWidth: number;
//   outChannels: number;
//   isChannelsFirst: boolean;
//   strideHeight: number;
//   strideWidth: number;
//   dilationHeight: number;
//   dilationWidth: number;
//   filterHeight: number;
//   filterWidth: number;
//   effectiveFilterHeight: number;
//   effectiveFilterWidth: number;
//   padInfo: PadInfo;
//   inShape: [number, number, number, number];
//   outShape: [number, number, number, number];
//   filterShape: [number, number, number, number];
// }

// const parseTupleParam = (param: number|number[]): [number, number, number] => {
//   if (typeof param === 'number') {
//     return [param, param, param];
//   }
//   if (param.length === 2) {
//     return [param[0], param[1], 1];
//   }
//   return param as [number, number, number];
// };

// /* See https://www.tensorflow.org/api_docs/python/tf/nn/atrous_conv2d
//  * Atrous convolution is equivalent to standard convolution with upsampled
//  * filters with effective_filter_height =
//  * filter_height + (filter_height - 1) * (dilation - 1)
//  * and effective_filter_width =
//  * filter_width + (filter_width - 1) * (dilation - 1),
//  * produced by inserting dilation - 1 zeros along consecutive elements across
//  * the filters' spatial dimensions.
//  * When there is a dilation, this converts a filter dimension to the
//  * effective filter dimension, so it can be used in a standard convolution.
//  */
// const getEffectiveFilterSize = (filterSize: number, dilation: number): number => {
//   if (dilation <= 1) {
//     return filterSize;
//   }

//   return filterSize + (filterSize - 1) * (dilation - 1);
// };


// /**
//  * Computes the information for a forward pass of a convolution/pooling
//  * operation.
//  */
// export const computeConv2DInfo =
//     (inShape: [number, number, number, number], filterShape: [number, number, number, number],
//      strides: number|[number, number], dilations: number|[number, number],
//      pad: 'SAME_UPPER'|'SAME_LOWER'|'VALID'|number|[number, number, number, number],
//      roundingMode: 'floor'|'round'|'ceil', depthwise: boolean, isChannelsFirst: boolean): Conv2DInfo => {
//       let [batchSize, inHeight, inWidth, inChannels] = [-1, -1, -1, -1];
//       if (isChannelsFirst) {
//         [batchSize, inChannels, inHeight, inWidth] = inShape;
//       } else {
//         [batchSize, inHeight, inWidth, inChannels] = inShape;
//       }

//       const [filterHeight, filterWidth, , filterChannels] = filterShape;
//       const [strideHeight, strideWidth] = parseTupleParam(strides);
//       const [dilationHeight, dilationWidth] = parseTupleParam(dilations);

//       const effectiveFilterHeight = getEffectiveFilterSize(filterHeight, dilationHeight);
//       const effectiveFilterWidth = getEffectiveFilterSize(filterWidth, dilationWidth);
//       const {padInfo, outHeight, outWidth} = getPadAndOutInfo(
//           pad, inHeight, inWidth, strideHeight, strideWidth, effectiveFilterHeight, effectiveFilterWidth,
//           roundingMode, dataFormat);

//       const outChannels = depthwise ? filterChannels * inChannels : filterChannels;

//       let outShape: [number, number, number, number];
//       if (dataFormat === 'channelsFirst') {
//         outShape = [batchSize, outChannels, outHeight, outWidth];
//       } else if (dataFormat === 'channelsLast') {
//         outShape = [batchSize, outHeight, outWidth, outChannels];
//       }

//       return {
//         batchSize,
//         dataFormat,
//         inHeight,
//         inWidth,
//         inChannels,
//         outHeight,
//         outWidth,
//         outChannels,
//         padInfo,
//         strideHeight,
//         strideWidth,
//         filterHeight,
//         filterWidth,
//         effectiveFilterHeight,
//         effectiveFilterWidth,
//         dilationHeight,
//         dilationWidth,
//         inShape,
//         outShape,
//         filterShape
//       };
//     }
