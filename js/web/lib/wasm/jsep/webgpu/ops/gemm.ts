// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../../../attribute-with-cache-key';
// import {Graph} from '../../../graph';
// import {OperatorAsyncImplementation, OperatorInitialization} from '../../../operators';
// import {Tensor} from '../../../tensor';
// import {GemmUtil, ShapeUtil} from '../../../util';
// import {WebGpuInferenceHandler} from '../inference-handler';
// import {GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

// import {WORKGROUP_SIZE} from './common';

// export interface GemmAttributes extends AttributeWithCacheKey {
//   transA: boolean;
//   transB: boolean;
//   alpha: number;
//   beta: number;
//   isOptionalC: boolean;  // in opset 11, C becomes optional
// }

// export const gemm: OperatorAsyncImplementation<GemmAttributes> = async(
//     inferenceHandler: WebGpuInferenceHandler, inputs: Tensor[], attributes: GemmAttributes): Promise<Tensor[]> => {
//   validateInputs(inputs, attributes);
//   return inferenceHandler.run(createGemmProgramInfoLoader(inputs, attributes), inputs);
// };

// const parseGemmAttributes = (node: Graph.Node, isOptionalC: boolean): GemmAttributes => {
//   const transA = node.attributes.getInt('transA', 0) !== 0;
//   const transB = node.attributes.getInt('transB', 0) !== 0;
//   const alpha = node.attributes.getFloat('alpha', 1.0);
//   const beta = node.attributes.getFloat('beta', 1.0);
//   return createAttributeWithCacheKey({transA, transB, alpha, beta, isOptionalC});
// };

// export const parseGemmAttributesV7: OperatorInitialization<GemmAttributes> = (node: Graph.Node): GemmAttributes =>
//     parseGemmAttributes(node, false);

// export const parseGemmAttributesV11: OperatorInitialization<GemmAttributes> = (node: Graph.Node): GemmAttributes =>
//     parseGemmAttributes(node, true);

// const createGemmProgramInfoLoader = (inputs: Tensor[], attributes: GemmAttributes): ProgramInfoLoader => {
//   const metadata = {
//     name: 'Gemm',
//     inputTypes: inputs.length === 3 ? [GpuDataType.default, GpuDataType.default, GpuDataType.default] :
//                                       [GpuDataType.default, GpuDataType.default],
//     cacheHint: attributes.cacheKey
//   };

//   return {...metadata, get: () => createGemmProgramInfo(metadata, inputs, attributes)};
// };

// const offsetC = (m: number, n: number, dims: readonly number[]): string => {
//   const broadcastM = (dims.length === 1 && m !== 1) || (dims.length === 2 && dims[0] !== m);
//   const broadcastN = dims[dims.length - 1] !== n;

//   let offset = '0u';
//   if (!broadcastM) {
//     offset += `+ m * ${dims[dims.length - 1]}u`;
//   }
//   if (!broadcastN) {
//     offset += '+n';
//   }

//   return offset;
// };

// const createGemmProgramInfo =
//     (metadata: ProgramMetadata, inputs: Tensor[], attributes: GemmAttributes): ProgramInfo => {
//       const aShape = inputs[0].dims.slice();
//       const bShape = inputs[1].dims.slice();
//       const [M, N, K] = GemmUtil.getShapeOfGemmResult(
//           aShape, attributes.transA, bShape, attributes.transB, inputs.length === 3 ? inputs[2].dims : undefined);
//       const outputShape = [M, N];
//       if (!outputShape) {
//         throw new Error('Can\'t use gemm on the given tensors');
//       }
//       const outputSize = ShapeUtil.size(outputShape);
//       let line = '';
//       if (attributes.transA && attributes.transB) {
//         line = 'value += a[k * M + m] * b[n * K + k];';
//       } else if (attributes.transA && !attributes.transB) {
//         line = 'value += a[k * M + m] * b[k * N + n];';
//       } else if (!attributes.transA && attributes.transB) {
//         line = 'value += a[m * K + k] * b[n * K + k];';
//       } else if (!attributes.transA && !attributes.transB) {
//         line = 'value += a[m * K + k] * b[k * N + n];';
//       }

//       const dataType = 'f32';  // TODO: support other data type
//       const calculateAlpha = attributes.alpha === 1 ? '' : 'value *= alpha;';
//       const calculateC = inputs.length === 3 ? `value += beta * c[${offsetC(M, N, inputs[2].dims)}];` : '';
//       const inputStorageBuffersDeclarations = [
//         `@group(0) @binding(0) var<storage, read> a : array<${dataType}>;`,
//         `@group(0) @binding(1) var<storage, read> b : array<${dataType}>;`
//       ];
//       if (inputs.length === 3) {
//         inputStorageBuffersDeclarations.push(`@group(0) @binding(2) var<storage, read> c : array<${dataType}>;`);
//       }
//       const shaderSource = `
//   const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;
//   const M: u32 = ${M}u;
//   const N: u32 = ${N}u;
//   const K: u32 = ${K}u;
//   const alpha = ${dataType}(${attributes.alpha});
//   const beta = ${dataType}(${attributes.beta});

//   ${inputStorageBuffersDeclarations.join('\n')}
//   @group(0) @binding(${inputs.length}) var<storage, read_write> output : array<${dataType}>;

//   @compute @workgroup_size(WORKGROUP_SIZE)
//   fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {

//     // Guard against out-of-bounds work group sizes
//     if (global_id.x >= ${outputSize}u) {
//       return;
//     }

//     let m = global_id.x / N;
//     let n = global_id.x % N;

//     var value = ${dataType}(0);
//     for (var k: u32 = 0u; k<${K}u; k++) {
//       ${line}
//     }

//     ${calculateAlpha}
//     ${calculateC}
//     output[global_id.x] = value;

//   }`;
//       return {
//         ...metadata,
//         outputs: [{dims: outputShape, type: inputs[0].type, gpuDataType: GpuDataType.default}],
//         shaderSource,
//         dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
//       };
//     };

// const validateInputs = (inputs: Tensor[], attributes: GemmAttributes): void => {
//   if (!inputs) {
//     throw new Error('Input is missing');
//   }
//   if (attributes.isOptionalC && (inputs.length < 2 || inputs.length > 3)) {
//     throw new Error('Invaid input shape.');
//   }
//   if (!attributes.isOptionalC && inputs.length !== 3) {
//     throw new Error('Gemm requires 3 inputs');
//   }

//   // 'C' can be of dimensionality 1 or 2 only
//   if (inputs.length === 3 && inputs[2].dims.length !== 1 && inputs[2].dims.length !== 2) {
//     throw new Error('Invalid input shape of C');
//   }

//   if ((inputs[0].type !== 'float32' && inputs[0].type !== 'float64') ||
//       (inputs[1].type !== 'float32' && inputs[1].type !== 'float64') ||
//       (inputs.length === 3 && inputs[2].type !== 'float32' && inputs[2].type !== 'float64')) {
//     throw new Error('Invalid input type.');
//   }

//   if ((inputs[0].type !== inputs[1].type) || (inputs.length === 3 && inputs[0].type !== inputs[2].type)) {
//     throw new Error('Input types are mismatched');
//   }
// };
