// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// import {Tensor} from '../../../tensor';
// import {ShapeUtil} from '../../../util';
// import {WebGpuInferenceHandler} from '../inference-handler';
// import {GpuDataType, ProgramInfo, ProgramMetadata} from '../types';

// import {WORKGROUP_SIZE} from './common';

// export const sum = async(inferenceHandler: WebGpuInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> => {
//   validateInputs(inputs);

//   const sumProgramMetadata = {name: 'Sum', inputTypes: new Array(inputs.length).fill(GpuDataType.default)};

//   return inferenceHandler.run(
//       {...sumProgramMetadata, get: () => createSumProgramInfo(inferenceHandler, inputs, sumProgramMetadata)},
//       inputs);
// };

// const createSumProgramInfo =
//     (inferenceHandler: WebGpuInferenceHandler, inputs: Tensor[], sumProgramMetadata: ProgramMetadata): ProgramInfo
//     => {
//       const dataType = 'f32';
//       const outputShape = inputs[0].dims;
//       const outputSize = ShapeUtil.size(outputShape);


//       const inputsDeclaration =
//           inputs.map((_, i) => `@group(0) @binding(${i}) var<storage, read> input${i} : array<${dataType}>;`);
//       const sumLine = inputs.map((_, i) => `input${i}[offset]`).join('+');
//       const shaderSource = `
//   const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;

//   ${inputsDeclaration.join('\n')}
//   @group(0) @binding(${inputs.length}) var<storage, read_write> output : array<${dataType}>;

//   @compute @workgroup_size(WORKGROUP_SIZE)
//   fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {

//     // Guard against out-of-bounds work group sizes
//     if (global_id.x >= ${outputSize}u) {
//       return;
//     }

//     let offset = global_id.x;

//     var value = ${dataType}(0);
//     value = ${sumLine};

//     output[offset] = value;
//   }`;
//       return {
//         ...sumProgramMetadata,
//         outputs: [{dims: outputShape, type: inputs[0].type, gpuDataType: GpuDataType.default}],
//         shaderSource,
//         dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
//       };
//     };

// const validateInputs = (inputs: Tensor[]): void => {
//   if (!inputs || inputs.length === 0) {
//     throw new Error('Sum requires inputs.');
//   }

//   const length = inputs[0].dims.length;
//   for (let i = 1; i < inputs.length; i++) {
//     if (length !== inputs[i].dims.length) {
//       throw new Error('Input shapes are mismatched. broadcasting not supported yet');
//     }

//     for (let j = 0; j < length; j++) {
//       if (inputs[0].dims[j] !== inputs[i].dims[j]) {
//         throw new Error('Input shapes are not matched. broadcasting not supported yet');
//       }
//     }
//   }

//   if (inputs[0].type !== 'float32' && inputs[0].type !== 'float64') {
//     throw new Error('Invalid input type.');
//   }
//   for (let i = 1; i < inputs.length; i++) {
//     if (inputs[0].type !== inputs[i].type) {
//       throw new Error('Input types are not matched.');
//     }
//   }
// };
