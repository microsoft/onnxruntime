// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../../../attribute-with-cache-key';
// import {Graph} from '../../../graph';
// import {OperatorAsyncImplementation, OperatorInitialization} from '../../../operators';
// import {Tensor} from '../../../tensor';
// import {PoolConvUtil, ShapeUtil} from '../../../util';
// import {WebGpuInferenceHandler} from '../inference-handler';
// import {GpuDataType, ProgramInfo, ProgramMetadata} from '../types';

// import {createIndicesHelper, WORKGROUP_SIZE} from './common';

// export interface AveragePoolAttributes extends AttributeWithCacheKey {
//   readonly autoPad: string;
//   readonly ceilMode: number;
//   readonly countIncludePad: boolean;
//   readonly kernelShape: readonly number[];
//   readonly strides: readonly number[];
//   readonly pads: readonly number[];
// }

// export const averagePool: OperatorAsyncImplementation<AveragePoolAttributes> =
//     async(inferenceHandler: WebGpuInferenceHandler, inputs: Tensor[], attributes: AveragePoolAttributes):
//         Promise<Tensor[]> => {
//           validateInputs(inputs);
//           const metadata = {name: 'AveragePool', inputTypes: [GpuDataType.default], cacheHint: attributes.cacheKey};
//           return inferenceHandler.run(
//               {...metadata, get: () => createAveragePoolProgramInfo(inputs, metadata, false, attributes)}, inputs);
//         };

// export const parseAveragePoolAttributes: OperatorInitialization<AveragePoolAttributes> =
//     (node: Graph.Node): AveragePoolAttributes => {
//       const autoPad = node.attributes.getString('auto_pad', 'NOTSET');
//       const ceilMode = node.attributes.getInt('ceil_mode', 0);
//       const countIncludePad = (node.attributes.getInt('count_include_pad', 0) === 0 ? false : true);
//       const kernelShape = node.attributes.getInts('kernel_shape');
//       const strides = node.attributes.getInts('strides', []);
//       const pads = node.attributes.getInts('pads', []);

//       // TODO: support attribute 'ceil_mode'
//       if (ceilMode !== 0) {
//         throw new Error('using ceil() in shape computation is not yet supported for AveragePool');
//       }

//       return createAttributeWithCacheKey({autoPad, ceilMode, countIncludePad, kernelShape, strides, pads});
//     };

// const createAveragePoolProgramInfo =
//     (inputs: Tensor[], metadata: ProgramMetadata, isGlobalOperator: boolean,
//      attributes: AveragePoolAttributes): ProgramInfo => {
//       const [adjustedAttributes, outputShape] =
//           getAdjustedPoolAttributesAndOutputShape(inputs, attributes, isGlobalOperator);
//       const kernelSize = ShapeUtil.size(adjustedAttributes.kernelShape);

//       const dataType = 'f32';

//       const op1 = 'value += x_val;';
//       let op2 = '';
//       if (adjustedAttributes.countIncludePad) {
//         op2 += `value /= ${dataType}(${kernelSize});`;
//       } else {
//         op2 += `value /= ${dataType}(${kernelSize} - pad);`;
//       }
//       return {
//         ...metadata,
//         outputs: [{dims: outputShape, type: inputs[0].type, gpuDataType: GpuDataType.default}],
//         shaderSource: generatePoolingCode(inputs[0].dims, outputShape, adjustedAttributes, op1, op2, dataType,
//         '0.0'), dispatchGroup: () => ({x: Math.ceil(ShapeUtil.size(outputShape) / 64 /* workgroup size */)})
//       };
//     };

// export const globalAveragePool: OperatorAsyncImplementation<AveragePoolAttributes> =
//     async(inferenceHandler: WebGpuInferenceHandler, inputs: Tensor[], attributes: AveragePoolAttributes):
//         Promise<Tensor[]> => {
//           validateInputs(inputs);
//           const metadata = {
//             name: 'GlobalAveragePool',
//             inputTypes: [GpuDataType.default],
//             cacheHint: `${attributes.countIncludePad}`
//           };
//           return inferenceHandler.run(
//               {...metadata, get: () => createAveragePoolProgramInfo(inputs, metadata, true, attributes)}, inputs);
//         };

// export const parseGlobalAveragePoolAttributes: OperatorInitialization<AveragePoolAttributes> =
//     (node: Graph.Node): AveragePoolAttributes => {
//       const countIncludePad = (node.attributes.getInt('count_include_pad', 0) === 0 ? false : true);
//       return createAttributeWithCacheKey(
//           {autoPad: '', ceilMode: 0, countIncludePad, kernelShape: [], strides: [], pads: []});
//     };

// export interface MaxPoolAttributes extends AveragePoolAttributes {
//   readonly storageOrder: number;
//   readonly dilations: number[];
// }

// export const maxPool: OperatorAsyncImplementation<MaxPoolAttributes> = async(
//     inferenceHandler: WebGpuInferenceHandler, inputs: Tensor[], attributes: MaxPoolAttributes): Promise<Tensor[]>
//     => {
//   validateInputs(inputs);
//   const metadata = {name: 'MaxPool', inputTypes: [GpuDataType.default], cacheHint: attributes.cacheKey};
//   return inferenceHandler.run(
//       {...metadata, get: () => createMaxPoolProgramInfo(inputs, metadata, false, attributes)}, inputs);
// };

// export const parseMaxPoolAttributes: OperatorInitialization<MaxPoolAttributes> =
//     (node: Graph.Node): MaxPoolAttributes => {
//       const autoPad = node.attributes.getString('auto_pad', 'NOTSET');
//       const ceilMode = node.attributes.getInt('ceil_mode', 0);
//       const kernelShape = node.attributes.getInts('kernel_shape');
//       const strides = node.attributes.getInts('strides', []);
//       const pads = node.attributes.getInts('pads', []);
//       const storageOrder = node.attributes.getInt('storage_order', 0);
//       const dilations = node.attributes.getInts('dilations', []);

//       // TODO: support attribute 'ceil_mode' and 'storage_order'
//       if (storageOrder !== 0) {
//         throw new Error('column major storage order is not yet supported for MaxPool');
//       }
//       if (ceilMode !== 0) {
//         throw new Error('using ceil() in shape computation is not yet supported for MaxPool');
//       }

//       return createAttributeWithCacheKey(
//           {autoPad, ceilMode, countIncludePad: false, kernelShape, strides, pads, storageOrder, dilations});
//     };

// const createMaxPoolProgramInfo =
//     (inputs: Tensor[], metadata: ProgramMetadata, isGlobalOperator: boolean, attributes: MaxPoolAttributes):
//         ProgramInfo => {
//           const [adjustedAttributes, outputShape] =
//               getAdjustedPoolAttributesAndOutputShape(inputs, attributes, isGlobalOperator);
//           const op1 = `
//       value = max(x_val, value);
//     `;
//           const op2 = '';
//           return {
//             ...metadata,
//             outputs: [{dims: outputShape, type: inputs[0].type, gpuDataType: GpuDataType.default}],
//             shaderSource: generatePoolingCode(inputs[0].dims, outputShape, adjustedAttributes, op1, op2, 'f32',
//             '-1e5'), dispatchGroup: () => ({x: Math.ceil(ShapeUtil.size(outputShape) / 64 /* workgroup size */)})
//           };
//         };

// const getAdjustedPoolAttributesAndOutputShape =
//     (inputs: Tensor[], attributes: AveragePoolAttributes|MaxPoolAttributes, isGlobalOperator: boolean):
//         [AveragePoolAttributes|MaxPoolAttributes, number[]] => {
//           const inputShape = inputs[0].dims.slice();
//           const hasDilations = Object.hasOwnProperty.call(attributes, 'dilations');
//           const kernelShape = attributes.kernelShape.slice();
//           const strides = attributes.strides.slice();
//           const dilations: number[] = hasDilations ? (attributes as MaxPoolAttributes).dilations.slice() : [];
//           const pads = attributes.pads.slice();
//           PoolConvUtil.adjustPoolAttributes(isGlobalOperator, inputShape, kernelShape, strides, dilations, pads);

//           const outputShape = PoolConvUtil.computePoolOutputShape(
//               isGlobalOperator, inputShape, strides, dilations, kernelShape, pads, attributes.autoPad);

//           const newAttributes = Object.assign({}, attributes);
//           if (hasDilations) {
//             Object.assign(newAttributes, {kernelShape, strides, pads, dilations, cacheKey: attributes.cacheKey});
//           } else {
//             Object.assign(newAttributes, {kernelShape, strides, pads, cacheKey: attributes.cacheKey});
//           }
//           return [newAttributes, outputShape];
//         };

// const globalMaxPoolAttributes = {
//   autoPad: '',
//   ceilMode: 0,
//   countIncludePad: false,
//   kernelShape: [],
//   strides: [],
//   pads: [],
//   storageOrder: 0,
//   dilations: [],
//   cacheKey: ''
// };

// const globalMaxPoolMetadata = {
//   name: 'GlobalMaxPool',
//   inputTypes: [GpuDataType.default]
// };

// export const globalMaxPool = async(inferenceHandler: WebGpuInferenceHandler, inputs: Tensor[]): Promise<Tensor[]>
// => {
//   validateInputs(inputs);
//   return inferenceHandler.run(
//       {
//         ...globalMaxPoolMetadata,
//         get: () => createMaxPoolProgramInfo(inputs, globalMaxPoolMetadata, true, globalMaxPoolAttributes)
//       },
//       inputs);
// };

// const validateInputs = (inputs: Tensor[]): void => {
//   if (!inputs || inputs.length !== 1) {
//     throw new Error('Pool ops requires 1 input.');
//   }
//   if (inputs[0].type !== 'float32' && inputs[0].type !== 'float64') {
//     throw new Error('Invalid input type.');
//   }
// };

// const generatePoolingCode =
//     (inputDims: readonly number[], outputShape: readonly number[], attributes: AveragePoolAttributes, op1: string,
//      op2: string, dataType: string, start: string): string => {
//       const rank = inputDims.length;
//       const outputSize = ShapeUtil.size(outputShape);
//       const outputIndicesHelper = createIndicesHelper('output', outputShape);
//       const xIndicesHelper = createIndicesHelper('x', inputDims);

//       if (attributes.kernelShape.length <= 2) {
//         const kw = attributes.kernelShape[attributes.kernelShape.length - 1];
//         const sw = attributes.strides[attributes.strides.length - 1];
//         const pwStart = attributes.pads[attributes.pads.length / 2 - 1];
//         const pwEnd = attributes.pads[attributes.pads.length - 1];
//         const dimW = inputDims[rank - 1];
//         let codeW = '';
//         let codeH = '';
//         let codeHEnd = '';
//         if (pwStart + pwEnd !== 0) {
//           codeW = `
//           for (var i: u32 = 0u; i < ${kw}u; i++) {
//             xIndices[${rank - 1}] = indices[${rank - 1}] * ${sw} - ${pwStart} + i;
//             if (xIndices[${rank - 1}] < 0 || xIndices[${rank - 1}] >= ${dimW}) {
//               pad++;
//               continue;
//             }
//             let x_val = x[${xIndicesHelper.i2oExpression('xIndices')}];
//             ${op1}
//           }`;
//         } else {
//           codeW = `
//           for (var i: u32 = 0u; i < ${kw}u; i++) {
//             xIndices[${rank - 1}] = indices[${rank - 1}] * ${sw} - ${pwStart} + i;
//             let x_val = x[${xIndicesHelper.i2oExpression('xIndices')}];
//             ${op1}
//           }`;
//         }

//         if (attributes.kernelShape.length === 2) {
//           const kh = attributes.kernelShape[attributes.kernelShape.length - 2];
//           const sh = attributes.strides[attributes.strides.length - 2];
//           const phStart = attributes.pads[attributes.pads.length / 2 - 2];
//           const phEnd = attributes.pads[attributes.pads.length - 2];
//           const dimH = inputDims[rank - 2];
//           if (phStart + phEnd !== 0) {
//             codeH = `
//             for (var j: u32 = 0u; j < ${kh}u; j++) {
//               xIndices[${rank - 2}] = indices[${rank - 2}] * ${sh} - ${phStart} + j;
//               if (xIndices[${rank - 2}] < 0 || xIndices[${rank - 2}] >= ${dimH}) {
//                 pad+= ${kw};
//                 continue;
//               }
//           `;
//           } else {
//             codeH = `
//             for (var j: u32 = 0u; j < ${kh}u; j++) {
//               xIndices[${rank - 2}] = indices[${rank - 2}] * ${sh} - ${phStart} + j;
//             `;
//           }
//           codeHEnd = `
//           }
//         `;
//         }

//         const poolingCode = `
//         const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;
//         @group(0) @binding(0) var<storage, read> x : array<${dataType}>;
//         @group(0) @binding(1) var<storage, read_write> output : array<${dataType}>;

//         ${outputIndicesHelper.o2iImpl}
//         ${xIndicesHelper.i2oImpl}

//         @compute @workgroup_size(WORKGROUP_SIZE)
//         fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {

//           // Guard against out-of-bounds work group sizes
//           if (global_id.x >= ${outputSize}u) {
//             return;
//           }

//           ${outputIndicesHelper.indicesVariableDeclaration('indices')}
//           ${outputIndicesHelper.o2iCall('global_id.x', 'indices')}
//           ${outputIndicesHelper.indicesVariableDeclaration('xIndices')}
//           ${outputIndicesHelper.o2iCall('global_id.x', 'xIndices')}

//           var value: ${dataType} = ${dataType}(${start});
//           var pad = 0;
//           ${codeH}
//           ${codeW}
//           ${codeHEnd}
//           ${op2}

//           output[global_id.x] = value;
//         }`;
//         return poolingCode;
//       } else {
//         const kernelSize = ShapeUtil.size(attributes.kernelShape);
//         const kernelStrides = ShapeUtil.computeStrides(attributes.kernelShape);
//         const stridesRank = kernelStrides.length;
//         const padsRank = attributes.pads.length;
//         const hasPads = attributes.pads.reduce((sum, cur) => sum + cur);
//         let padCode = '';
//         if (hasPads) {
//           padCode = `
//             if (xIndices[j] >= inputDims[j]) {
//               pad++;
//               isPad = true;
//               break;
//             }
//           }
//           if (!isPad) {
//             let x_val = x[${xIndicesHelper.i2oExpression('xIndices')}];
//             ${op1}
//           }`;
//         } else {
//           padCode = `
//           }
//           let x_val = x[${xIndicesHelper.i2oExpression('xIndices')}];
//           ${op1}
//         `;
//         }
//         const poolingCode = `
//         const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;
//         @group(0) @binding(0) var<storage, read> x : array<${dataType}>;
//         @group(0) @binding(1) var<storage, read_write> output : array<${dataType}>;

//         ${outputIndicesHelper.o2iImpl}
//         ${xIndicesHelper.i2oImpl}

//         const pads = array<u32, ${padsRank}>(${attributes.pads.map(i => `${i}u`).join(',')});
//         const inputDims = array<u32, ${rank}>(${inputDims.map(i => `${i}u`).join(',')});
//         const kernelStrides = array<u32, ${stridesRank}>(${kernelStrides.map(i => `${i}u`).join(',')});
//         const strides = array<u32, ${stridesRank}>(${attributes.strides.map(i => `${i}u`).join(',')});

//         @compute @workgroup_size(WORKGROUP_SIZE)
//         fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {

//           // Guard against out-of-bounds work group sizes
//           if (global_id.x >= ${outputSize}u) {
//             return;
//           }

//           ${outputIndicesHelper.indicesVariableDeclaration('indices')}
//           ${outputIndicesHelper.o2iCall('global_id.x', 'indices')}
//           ${outputIndicesHelper.indicesVariableDeclaration('xIndices')}
//           ${outputIndicesHelper.o2iCall('global_id.x', 'xIndices')}

//           var offsets: array<u32, ${stridesRank}>;

//           var value = ${dataType}(${start});
//           var pad = 0;
//           var isPad = false;

//           for (var i: u32 = 0u; i < ${kernelSize}u; i++) {
//             var offset = i;
//             for (var j = 0u; j < ${stridesRank - 1}u; j++) {
//               offsets[j] = offset / kernelStrides[j];
//               offset -= offsets[j] * kernelStrides[j];
//             }
//             offsets[${stridesRank - 1}] = offset;

//             isPad = false;
//             for (var j = ${rank - stridesRank}u; j < ${rank}u; j++) {
//               xIndices[j] = indices[j] * strides[j - ${rank - stridesRank}u]
//                 + offsets[j - ${rank - stridesRank}u] - pads[j - 2u];
//               ${padCode}
//           }
//           ${op2}

//           output[global_id.x] = value;
//         }`;
//         return poolingCode;
//       }
//     };
