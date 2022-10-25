// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// import {Logger} from '../../../instrument';
// import {Tensor} from '../../../tensor';
// import {ShapeUtil} from '../../../util';
// import {WebGpuInferenceHandler} from '../inference-handler';
// import {GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

// import {createIndicesHelper, WORKGROUP_SIZE} from './common';
// import {calculateOutputShape, ConvAttributes} from './conv';
// import {getActicationSnippet} from './fuse-utils';

// const createGroupedConvProgramMetadata = (hasBias: boolean, cacheHint: string): ProgramMetadata => ({
//   name: 'GroupedConv',
//   inputTypes: hasBias ? [GpuDataType.default, GpuDataType.default, GpuDataType.default] :
//                         [GpuDataType.default, GpuDataType.default],
//   cacheHint
// });

// const createGroupedConvProgramInfo =
//     (inferenceHandler: WebGpuInferenceHandler, inputs: readonly Tensor[], metadata: ProgramMetadata,
//      attributes: ConvAttributes): ProgramInfo => {
//       const hasBias = inputs.length > 2;
//       const processBias = hasBias ? 'value += b[output_channel];' : '';
//       const xShape = inputs[0].dims;
//       const wShape = inputs[1].dims;
//       const outputChannelsPerGroup = wShape[0] / attributes.group;

//       const dataType = 'f32';  // TODO: support other data type
//       const {activationFunction, applyActivation} = getActicationSnippet(attributes);
//       const inputStorageBuffersDeclarations = [
//         `@group(0) @binding(0) var<storage, read> x : array<${dataType}>;`,
//         `@group(0) @binding(1) var<storage, read> w : array<${dataType}>;`
//       ];
//       if (hasBias) {
//         inputStorageBuffersDeclarations.push(`@group(0) @binding(2) var<storage, read> b : array<${dataType}>;`);
//       }

//       Logger.verbose(
//           'GroupedConv',
//           `autpPad:${attributes.autoPad}, dilations:${attributes.dilations}, group:${attributes.group},
//           kernelShape:${
//               attributes.kernelShape}, pads:${attributes.pads}, strides:${attributes.strides}`);
//       const outputShape =
//           calculateOutputShape(xShape, wShape, attributes.dilations, attributes.pads, attributes.strides);
//       const outputSize = ShapeUtil.size(outputShape);
//       const outputIndicesHelper = createIndicesHelper('output', outputShape);
//       const xIndicesHelper = createIndicesHelper('x', xShape);
//       const wIndicesHelper = createIndicesHelper('w', wShape);

//       const shaderSource = `
//   const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;
//   const strides: vec2<u32> = vec2(${attributes.strides[0]}u, ${attributes.strides[1]}u);
//   const pads: vec2<u32> = vec2(${attributes.pads[0]}u, ${attributes.pads[1]}u);

//   ${inputStorageBuffersDeclarations.join('\n')}
//   @group(0) @binding(${inputStorageBuffersDeclarations.length}) var<storage, read_write> output : array<${dataType}>;

//   ${activationFunction}
//   ${outputIndicesHelper.o2iImpl}
//   ${xIndicesHelper.i2oImpl}
//   ${wIndicesHelper.i2oImpl}

//   @compute @workgroup_size(WORKGROUP_SIZE)
//   fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
//     // Guard against out-of-bounds work group sizes
//     if (global_id.x >= ${outputSize}u) {
//       return;
//     }

//     ${outputIndicesHelper.indicesVariableDeclaration('outputIndices')}
//     ${outputIndicesHelper.o2iCall('global_id.x', 'outputIndices')}
//     let batch: u32 = outputIndices[0];
//     let output_channel: u32 = outputIndices[1];
//     let xRCCorner: vec2<u32> = vec2<u32>(outputIndices[2], outputIndices[3]) * strides - pads;
//     let group_id: u32 = output_channel / ${outputChannelsPerGroup}u;

//     var value: ${dataType} = ${dataType}(0);
//     for (var wInChannel: u32 = 0u; wInChannel < ${wShape[1]}u; wInChannel++) {
//       let input_channel = group_id * ${wShape[1]}u + wInChannel;
//       for (var wHeight: u32 = 0u; wHeight < ${wShape[2]}u; wHeight++) {
//         let xHeight = xRCCorner.x + wHeight * ${attributes.dilations[0]}u;

//         if (xHeight < 0u || xHeight >= ${xShape[2]}u) {
//           continue;
//         }

//         for (var wWidth: u32 = 0u; wWidth < ${wShape[3]}u; wWidth++) {
//           let xWidth = xRCCorner.y + wWidth * ${attributes.dilations[1]}u;
//           if (xWidth < 0u || xWidth >= ${xShape[3]}u) {
//             continue;
//           }

//           ${
//           xIndicesHelper.indicesVariableDeclaration(
//               'xIndices',
//               [
//                 'batch', 'input_channel', 'xHeight', 'xWidth'
//               ])}
//           let xVal = x[${xIndicesHelper.i2oExpression('xIndices')}];
//           ${
//           wIndicesHelper.indicesVariableDeclaration('wIndices', [
//             'output_channel', 'wInChannel', 'wHeight', 'wWidth'
//           ])}
//           let wVal = w[${wIndicesHelper.i2oExpression('wIndices')}];
//           value += xVal*wVal;
//         }
//       }
//     }
//     ${processBias}
//     ${applyActivation}
//     output[global_id.x] = value;
//   }`;
//       return {
//         ...metadata,
//         outputs: [{dims: outputShape, type: inputs[0].type, gpuDataType: GpuDataType.default}],
//         shaderSource,
//         dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
//       };
//     };

// export const createGroupedConvProgramInfoLoader =
//     (inferenceHandler: WebGpuInferenceHandler, inputs: readonly Tensor[], attributes: ConvAttributes):
//         ProgramInfoLoader => {
//           const metadata = createGroupedConvProgramMetadata(inputs.length > 2, attributes.cacheKey);
//           return {...metadata, get: () => createGroupedConvProgramInfo(inferenceHandler, inputs, metadata,
//           attributes)};
//         };
