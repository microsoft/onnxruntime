// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../../../attribute-with-cache-key';
// import {Graph} from '../../../graph';
// import {NUMBER_TYPES, OperatorInitialization} from '../../../operators';
// import {Tensor} from '../../../tensor';
// import {ShapeUtil} from '../../../util';
// import {WebGpuInferenceHandler} from '../inference-handler';
// import {GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

// import {createIndicesHelper, WORKGROUP_SIZE} from './common';

// interface GatherAttributes extends AttributeWithCacheKey {
//   readonly axis: number;
// }

// export const gather = async(
//     inferenceHandler: WebGpuInferenceHandler, inputs: Tensor[], attributes: GatherAttributes): Promise<Tensor[]> => {
//   validateInputs(inputs, attributes.axis);
//   return inferenceHandler.run(createGatherProgramInfoLoader(inputs, attributes), inputs);
// };

// export const parseGatherAttributes: OperatorInitialization<GatherAttributes> = (node: Graph.Node): GatherAttributes
// =>
//     createAttributeWithCacheKey({axis: node.attributes.getInt('axis', 0)});

// const gatherProgramMetadata = {
//   name: 'Gather',
//   inputTypes: [GpuDataType.default, GpuDataType.default]
// };

// const createGatherProgramInfo =
//     (metadata: ProgramMetadata, inputs: Tensor[], axis: number, dataType = 'f32'): ProgramInfo => {
//       const dataShape = inputs[0].dims.slice();
//       const indicesShape = inputs[1].dims.slice();
//       const outputShape = new Array(dataShape.length + indicesShape.length - 1);

//       axis = ShapeUtil.normalizeAxis(axis, dataShape.length);
//       const indexCopyOps: string[] = [];
//       if (indicesShape.length > 1) {
//         indexCopyOps.push('indicesIdx[0] = 0u;');
//       } else {
//         indexCopyOps.push('indicesIdx = 0u;');
//       }
//       for (let i = 0; i < outputShape.length; i++) {
//         // outputShape is divided into three parts: A, B, C
//         // |0        axis|  axis + indicesShape.length |          end|
//         // |     A       |             B               |      C      |
//         //
//         // dataIdx: [A, inputs[1][B], C]
//         const outputIdxLValue = outputShape.length > 1 ? `outputIdx[${i}]` : 'outputIdx';
//         if (i < axis) {  // A
//           const dataIdxLValue = dataShape.length > 1 ? `dataIdx[${i}]` : 'dataIdx';
//           outputShape[i] = dataShape[i];
//           indexCopyOps.push(`${dataIdxLValue} = ${outputIdxLValue};`);
//         } else {
//           if (i < axis + indicesShape.length) {  // B
//             const indicesIdxLValue = indicesShape.length > 1 ? `indicesIdx[${i - axis}]` : 'indicesIdx';
//             outputShape[i] = indicesShape[i - axis];
//             indexCopyOps.push(`${indicesIdxLValue} = ${outputIdxLValue};`);
//           } else {  // C
//             const dataIdxLValue = dataShape.length > 1 ? `dataIdx[${i - indicesShape.length + 1}]` : 'dataIdx';
//             outputShape[i] = dataShape[i - indicesShape.length + 1];  // skip 1 for axis
//             indexCopyOps.push(`${dataIdxLValue} = ${outputIdxLValue};`);
//           }
//         }
//       }
//       const outputSize = ShapeUtil.size(outputShape);
//       const outputIndicesHelper = createIndicesHelper('output', outputShape);
//       const dataIndicesHelper = createIndicesHelper('data', dataShape);
//       const indicesIndicesHelper = createIndicesHelper('indices', indicesShape);

//       const shaderSource = `
//     const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;

//     @group(0) @binding(0) var<storage, read> data : array<${dataType}>;
//     @group(0) @binding(1) var<storage, read> indices : array<i32>;
//     @group(0) @binding(2) var<storage, read_write> output : array<${dataType}>;

//     ${outputIndicesHelper.o2iImpl}
//     ${indicesIndicesHelper.i2oImpl}
//     ${dataIndicesHelper.i2oImpl}

//     @compute @workgroup_size(WORKGROUP_SIZE)
//     fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {

//       // Guard against out-of-bounds work group sizes
//       if (global_id.x >= ${outputSize}u) {
//         return;
//       }

//       ${outputIndicesHelper.indicesVariableDeclaration('outputIdx')}
//       ${outputIndicesHelper.o2iCall('global_id.x', 'outputIdx')}
//       ${dataIndicesHelper.indicesVariableDeclaration('dataIdx')}
//       ${indicesIndicesHelper.indicesVariableDeclaration('indicesIdx')}
//       ${indexCopyOps.join('\n        ')}
//       let idx = indices[${indicesIndicesHelper.i2oExpression('indicesIdx')}];
//       dataIdx${dataShape.length > 1 ? `[${axis}]` : ''} = u32(select(idx, idx + ${dataShape[axis]}, idx < 0));
//       output[global_id.x] = data[${dataIndicesHelper.i2oExpression('dataIdx')}];
//     }`;
//       return {
//         ...metadata,
//         outputs: [{dims: outputShape, type: inputs[0].type, gpuDataType: GpuDataType.default}],
//         shaderSource,
//         dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
//       };
//     };

// const createGatherProgramInfoLoader = (inputs: Tensor[], attributes: GatherAttributes): ProgramInfoLoader => {
//   const metadata = {...gatherProgramMetadata, cacheHint: attributes.cacheKey};
//   return {...metadata, get: () => createGatherProgramInfo(metadata, inputs, attributes.axis)};
// };

// const validateInputs = (inputs: Tensor[], axis: number): void => {
//   if (!inputs || inputs.length !== 2) {
//     throw new Error('Gather requires 2 inputs.');
//   }
//   const tensorRank = inputs[0].dims.length;
//   if (tensorRank < 1) {
//     throw new Error('Invalid input shape.');
//   }
//   if (axis < -tensorRank || axis > tensorRank - 1) {
//     throw new Error('Invalid axis.');
//   }
//   if (NUMBER_TYPES.indexOf(inputs[0].type) === -1) {
//     throw new Error('Invaid input type.');
//   }
//   if (inputs[1].type !== 'int32') {
//     throw new Error('Invaid input type.');
//   }
// };
