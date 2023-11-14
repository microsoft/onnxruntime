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

// sampled from [@tensorflow/tfjs] tfjs-backend-webgpu/src/matmul_packed_webgpu.ts
//
// modified to fit the needs of the project

import {TensorView} from '../../../tensor-view';
import {ShapeUtil} from '../../../util';
import {ProgramInfo, ProgramInputTensorInfoDependency, ProgramUniform} from '../../types';
import {createTensorShapeVariables, enableShapesUniforms, getBroadcastDims, IndicesHelper, inputVariable, outputVariable, ShaderHelper, tensorTypeToWsglStorageType} from '../common';
import {getActivationSnippet, InternalActivationAttributes} from '../fuse-utils';

import {typeSnippet} from './activation_util';

const writeDataToSubAVec4Snippet = (transpose: boolean, batchDims?: IndicesHelper) => {
  if (transpose) {
    return `
        mm_Asub[inputRow][inputCol] = mm_readA(batch,
          kStart + inputRow,
          globalRowStart / innerElementSize + inputCol${batchDims ? ', batchIndices' : ''});
        `;

  } else {
    return `
        mm_Asub[inputRow][inputCol] = mm_readA(batch,
          globalRow + innerRow,
          kStart / innerElementSize + inputCol${batchDims ? ', batchIndices' : ''});
        `;
  }
};

const calculateResultSnippet = (transposeA: boolean, innerElementSize: number) => {
  if (transposeA) {
    return `
        let ACached0 = mm_Asub[k * innerElementSize][localRow];
        let ACached1 = mm_Asub[k * innerElementSize + 1][localRow];
        let ACached2 = mm_Asub[k * innerElementSize + 2][localRow];
        ${innerElementSize === 3 ? '' : 'let ACached3 = mm_Asub[k * innerElementSize + 3][localRow];'}
        for (var i = 0; i < rowPerThread; i = i + 1) {
          acc[i] = BCached0 * ACached0[i] + acc[i];
          acc[i] = BCached1 * ACached1[i] + acc[i];
          acc[i] = BCached2 * ACached2[i] + acc[i];
          ${innerElementSize === 3 ? '' : 'acc[i] = BCached3 * ACached3[i] + acc[i];'}
        }`;
  } else {
    return `
        for (var i = 0; i < rowPerThread; i = i + 1) {
          let ACached = mm_Asub[tileRow + i][k];
          acc[i] = BCached0 * ACached.x + acc[i];
          acc[i] = BCached1 * ACached.y + acc[i];
          acc[i] = BCached2 * ACached.z + acc[i];
          ${innerElementSize === 3 ? '' : 'acc[i] = BCached3 * ACached.w + acc[i];'}
        }`;
  }
};

export const makeMatMulPackedVec4Source =
    (workPerThread: number[], workgroupSize: [number, number, number], type = 'f32', batchDims?: IndicesHelper,
     transposeA = false, tileInner = 32, splitK = false, splitedDimInner = 32): string => {
      const tileAOuter = workgroupSize[1] * workPerThread[1];
      const tileBOuter = workgroupSize[0] * workPerThread[0];
      const tileAWidth = transposeA ? tileAOuter : tileInner;
      const tileAHight = transposeA ? tileInner : tileAOuter;
      const innerElementSize = tileAWidth / workgroupSize[0];
      const rowPerThreadB = tileInner / workgroupSize[1];

      if (!(((transposeA && innerElementSize === 4 && workPerThread[1] === 4) ||
             (!transposeA && (innerElementSize === 3 || innerElementSize === 4))) &&
            tileAWidth % workgroupSize[0] === 0 && tileInner % workgroupSize[1] === 0 && workPerThread[0] === 4)) {
        throw new Error(`If transposeA ${transposeA} is true, innerElementSize ${
            innerElementSize} and workPerThread[1] ${workPerThread[1]} must be 4.
      Otherwise, innerElementSize ${innerElementSize} must be 3 or 4.
  tileAWidth ${tileAWidth} must be divisible by workgroupSize[0]${workgroupSize[0]}. tileInner ${
            tileInner} must be divisible by workgroupSize[1] ${workgroupSize[1]}. colPerThread ${
            workPerThread[0]} must be 4.`);
      }
      return `
var<workgroup> mm_Asub: array<array<vec${innerElementSize}<${type}>, ${tileAWidth / innerElementSize}>, ${tileAHight}>;
var<workgroup> mm_Bsub: array<array<vec4<${type}>, ${tileBOuter / workPerThread[0]}>, ${tileInner}>;

const rowPerThread = ${workPerThread[1]};
const colPerThread = ${workPerThread[0]};
const innerElementSize = ${innerElementSize};
const tileInner = ${tileInner};

@compute @workgroup_size(${workgroupSize[0]}, ${workgroupSize[1]}, ${workgroupSize[2]})
fn main(@builtin(local_invocation_id) localId : vec3<u32>,
        @builtin(global_invocation_id) globalId : vec3<u32>,
        @builtin(workgroup_id) workgroupId : vec3<u32>) {
  let localRow = i32(localId.y);
  let tileRow = localRow * rowPerThread;
  let tileCol = i32(localId.x);

  let globalRow =i32(globalId.y) * rowPerThread;
  let globalCol = i32(globalId.x);
  let batch = ${splitK ? '0' : 'i32(globalId.z)'};
  ${batchDims ? `let batchIndices = ${batchDims.offsetToIndices('u32(batch)')};` : ''}
  let globalRowStart = i32(workgroupId.y) * ${tileAOuter};

  let numTiles = ${splitK ? `${Math.ceil(splitedDimInner / tileInner)}` : '(uniforms.dimInner - 1) / tileInner + 1'};
  var kStart = ${splitK ? `i32(globalId.z) * ${splitedDimInner}` : '0'};

  var acc: array<vec4<${type}>, rowPerThread>;

  // Loop over shared dimension.
  let tileRowB = localRow * ${rowPerThreadB};
  for (var t = 0; t < numTiles; t = t + 1) {
      // Load one tile of A into local memory.
      for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
          let inputRow = tileRow + innerRow;
          let inputCol = tileCol;
          ${writeDataToSubAVec4Snippet(transposeA, batchDims)}
      }

      // Load one tile of B into local memory.
      for (var innerRow = 0; innerRow < ${rowPerThreadB}; innerRow = innerRow + 1) {
          let inputRow = tileRowB + innerRow;
          let inputCol = tileCol;
          mm_Bsub[inputRow][inputCol] = mm_readB(batch, kStart + inputRow, globalCol${
          batchDims ? ', batchIndices' : ''});
      }
      kStart = kStart + tileInner;
      workgroupBarrier();

      // Compute acc values for a single thread.
      for (var k = 0; k < tileInner / innerElementSize; k = k + 1) {
          let BCached0 = mm_Bsub[k * innerElementSize][tileCol];
          let BCached1 = mm_Bsub[k * innerElementSize + 1][tileCol];
          let BCached2 = mm_Bsub[k * innerElementSize + 2][tileCol];
          ${innerElementSize === 3 ? '' : 'let BCached3 = mm_Bsub[k * innerElementSize + 3][tileCol];'}

          ${calculateResultSnippet(transposeA, innerElementSize)}
      }

      workgroupBarrier();
  }

  for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
      mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);
  }
}`;
    };

const writeDataToSubASnippet = (transpose: boolean, batchDims?: IndicesHelper) => {
  if (transpose) {
    return `
            mm_Asub[inputRow][inputCol] = mm_readA(batch,
              kStart + inputRow,
              globalRowStart + inputCol${batchDims ? ', batchIndices' : ''});
            `;

  } else {
    return `
            mm_Asub[inputRow][inputCol] = mm_readA(batch,
              globalRowStart + inputRow,
              kStart + inputCol${batchDims ? ', batchIndices' : ''});
            `;
  }
};

const readDataFromSubASnippet = (transposeA: boolean) =>
    transposeA ? 'let ACached = mm_Asub[k][tileRow + innerRow];' : 'let ACached = mm_Asub[tileRow + innerRow][k];';

// sequentialAccessByThreads means sequential data in memory is accessed by
// threads, instead of a single thread (default behavior).
export const makeMatMulPackedSource =
    (workPerThread: number[], workgroupSize: [number, number, number], type = 'f32', batchDims?: IndicesHelper,
     transposeA = false, tileInner = 32, splitK = false, splitedDimInner = 32,
     sequentialAccessByThreads = false): string => {
      const tileAOuter = workPerThread[1] * workgroupSize[1];
      const tileBOuter = workPerThread[0] * workgroupSize[0];
      const tileAWidth = transposeA ? tileAOuter : tileInner;
      const tileAHight = transposeA ? tileInner : tileAOuter;

      if (!(tileAHight % workgroupSize[1] === 0 && tileAWidth % workgroupSize[0] === 0 &&
            tileInner % workgroupSize[1] === 0)) {
        throw new Error(`tileAHight ${tileAHight} must be divisible by workgroupSize[1]${
            workgroupSize[1]}, tileAWidth ${tileAWidth} must be divisible by workgroupSize[0]${
            workgroupSize[0]}, tileInner ${tileInner} must be divisible by workgroupSize[1]${workgroupSize[1]}`);
      }
      const rowPerThreadA = tileAHight / workgroupSize[1];
      const colPerThreadA = tileAWidth / workgroupSize[0];
      const rowPerThreadB = tileInner / workgroupSize[1];
      const matmulSnippet = sequentialAccessByThreads ?
          `
    let localRow = i32(localId.y);
    let localCol = i32(localId.x);
    let globalRowStart = i32(workgroupId.y) * ${tileAOuter};
    let globalColStart = i32(workgroupId.x) * ${tileBOuter};

    // Loop over shared dimension.
    for (var t = 0; t < numTiles; t = t + 1) {
      // Load one tile of A into local memory.
      for (var inputRow = localRow; inputRow < ${tileAHight}; inputRow = inputRow + ${workgroupSize[1]}) {
        for (var inputCol = localCol; inputCol < ${tileAWidth}; inputCol = inputCol + ${workgroupSize[0]}) {
          ${writeDataToSubASnippet(transposeA, batchDims)}
        }
      }
      // Load one tile of B into local memory.
      for (var inputRow = localRow; inputRow < ${tileInner}; inputRow = inputRow + ${workgroupSize[1]}) {
            for (var inputCol = localCol; inputCol < ${tileBOuter}; inputCol = inputCol + ${workgroupSize[0]}) {
          mm_Bsub[inputRow][inputCol] = mm_readB(batch,
            kStart + inputRow,
            globalColStart + inputCol${batchDims ? ', batchIndices' : ''});
        }
      }
      kStart = kStart + tileInner;
      workgroupBarrier();

      // Compute acc values for a single thread.
      var BCached : array<${type}, colPerThread>;
      for (var k = 0; k < tileInner; k = k + 1) {
        for (var inner = 0; inner < colPerThread; inner = inner + 1) {
          BCached[inner] = mm_Bsub[k][localCol + inner * ${workgroupSize[0]}];
        }
        for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
          let ACached = ${
              transposeA ? `mm_Asub[k][localRow + innerRow * ${workgroupSize[1]}];` :
                           `mm_Asub[localRow + innerRow * ${workgroupSize[1]}][k];`}
          for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {
            acc[innerRow][innerCol] = acc[innerRow][innerCol] +
                ACached * BCached[innerCol];
          }
        }
      }
      workgroupBarrier();
    }
    for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
      let gRow = globalRowStart + localRow + innerRow * ${workgroupSize[1]};
      for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {
        let gCol = globalColStart + localCol + innerCol * ${workgroupSize[0]};
        mm_write(batch, gRow, gCol, acc[innerRow][innerCol]);
      }
    }
    ` :
          `
let tileRow = i32(localId.y) * rowPerThread;
let tileCol = i32(localId.x) * colPerThread;

let globalRow = i32(globalId.y) * rowPerThread;
let globalCol = i32(globalId.x) * colPerThread;
let globalRowStart = i32(workgroupId.y) * ${tileAOuter};

let tileRowA = i32(localId.y) * ${rowPerThreadA};
let tileColA = i32(localId.x) * ${colPerThreadA};
let tileRowB = i32(localId.y) * ${rowPerThreadB};
// Loop over shared dimension.
for (var t = 0; t < numTiles; t = t + 1) {
  // Load one tile of A into local memory.
  for (var innerRow = 0; innerRow < ${rowPerThreadA}; innerRow = innerRow + 1) {
    for (var innerCol = 0; innerCol < ${colPerThreadA}; innerCol = innerCol + 1) {
      let inputRow = tileRowA + innerRow;
      let inputCol = tileColA + innerCol;
      ${writeDataToSubASnippet(transposeA, batchDims)}
    }
  }

  // Load one tile of B into local memory.
  for (var innerRow = 0; innerRow < ${rowPerThreadB}; innerRow = innerRow + 1) {
    for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {
      let inputRow = tileRowB + innerRow;
      let inputCol = tileCol + innerCol;
      mm_Bsub[inputRow][inputCol] = mm_readB(batch,
        kStart + inputRow,
        globalCol + innerCol${batchDims ? ', batchIndices' : ''});
    }
  }
  kStart = kStart + tileInner;
  workgroupBarrier();

  // Compute acc values for a single thread.
  var BCached : array<${type}, colPerThread>;
  for (var k = 0; k < tileInner; k = k + 1) {
    for (var inner = 0; inner < colPerThread; inner = inner + 1) {
      BCached[inner] = mm_Bsub[k][tileCol + inner];
    }

    for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
      ${readDataFromSubASnippet(transposeA)}
      for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {
        acc[innerRow][innerCol] = acc[innerRow][innerCol] + ACached * BCached[innerCol];
      }
    }
  }

  workgroupBarrier();
}

for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
  for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {
    mm_write(batch, globalRow + innerRow, globalCol + innerCol,
        acc[innerRow][innerCol]);
  }
}
`;

      return `
  var<workgroup> mm_Asub : array<array<${type}, ${tileAWidth}>, ${tileAHight}>;
  var<workgroup> mm_Bsub : array<array<${type}, ${tileBOuter}>, ${tileInner}>;
  const rowPerThread = ${workPerThread[1]};
  const colPerThread = ${workPerThread[0]};
  const tileInner = ${tileInner};

@compute @workgroup_size(${workgroupSize[0]}, ${workgroupSize[1]}, ${workgroupSize[2]})
fn main(@builtin(local_invocation_id) localId : vec3<u32>,
        @builtin(global_invocation_id) globalId : vec3<u32>,
        @builtin(workgroup_id) workgroupId : vec3<u32>) {
    let batch = ${splitK ? '0' : 'i32(globalId.z)'};
    ${batchDims ? `let batchIndices = ${batchDims.offsetToIndices('u32(batch)')};` : ''}
    let numTiles = ${splitK ? `${Math.ceil(splitedDimInner / tileInner)}` : '(uniforms.dimInner - 1) / tileInner + 1'};
    var kStart = ${splitK ? `i32(globalId.z) * ${splitedDimInner}` : '0'};

    var acc : array<array<${type}, colPerThread>, rowPerThread>;

    // Without this initialization strange values show up in acc.
    for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
      for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {
        acc[innerRow][innerCol] = 0.0;
      }
    }
    ${matmulSnippet}
  }
`;
    };

const matMulReadWriteFnSource =
    (component: number, hasBias: boolean, applyActivation: string, variables: IndicesHelper[],
     batchShapes: Array<readonly number[]>, isChannelsLast = false): string => {
      const batchAShape = batchShapes[0];
      const batchBShape = batchShapes[1];
      const batchShape = batchShapes[2];
      const batchVariable = variables[0];
      const aVariable = variables[1];
      const bVariable = variables[2];
      const outputVariable = variables[3];
      const broadCastADims = getBroadcastDims(batchAShape, batchShape);
      const broadCastBDims = getBroadcastDims(batchBShape, batchShape);
      const dataType = tensorTypeToWsglStorageType(variables[0].type.tensor);
      const getAIndices = () => {
        const aRank = aVariable.rank;
        const batchRank = batchVariable.rank;
        let resStr = `var aIndices: ${aVariable.type.indices};`;
        for (let i = aRank - 2 - 1, j = batchRank - 1; i >= 0; i--, j--) {
          resStr += `\naIndices[${i}] = ${batchRank > 1 ? `batchIndices[${j}]` : 'batchIndices'};`;
        }
        broadCastADims.forEach(i => {
          resStr += `\naIndices[${i}] = 0;`;
        });
        resStr += `\naIndices[${aRank - 2}] = u32(row);
                   aIndices[${aRank - 1}] = u32(colIn);`;
        return resStr;
      };
      const getBIndices = () => {
        const bRank = bVariable.rank;
        const batchRank = batchVariable.rank;
        let resStr = `var bIndices: ${bVariable.type.indices};`;
        for (let i = bRank - 2 - 1, j = batchRank - 1; i >= 0; i--, j--) {
          resStr += `\nbIndices[${i}] = ${batchRank > 1 ? `batchIndices[${j}]` : 'batchIndices'};`;
        }
        broadCastBDims.forEach(i => {
          resStr += `\nbIndices[${i}] = 0;`;
        });
        resStr += `\nbIndices[${bRank - 2}] = u32(row);
                   bIndices[${bRank - 1}] = u32(colIn);`;
        return resStr;
      };
      const source = `
    fn mm_readA(batch: i32, row: i32, colIn: i32, batchIndices: ${batchVariable.type.indices}) -> ${
          typeSnippet(component, dataType)} {
      var value = ${typeSnippet(component, dataType)}(0.0);
      let col = colIn * ${component};
      if(row < uniforms.dimAOuter && col < uniforms.dimInner)
      {
        ${getAIndices()}
        value = ${aVariable.getByIndices('aIndices')};
      }
      return value;
    }

    fn mm_readB(batch: i32, row: i32, colIn: i32, batchIndices: ${batchVariable.type.indices}) -> ${
          typeSnippet(component, dataType)} {
      var value = ${typeSnippet(component, dataType)}(0.0);
      let col = colIn * ${component};
      if(row < uniforms.dimInner && col < uniforms.dimBOuter)
      {
        ${getBIndices()}
        value = ${bVariable.getByIndices('bIndices')};
      }
      return value;
    }

    fn mm_write(batch: i32, row: i32, colIn: i32, valueIn: ${typeSnippet(component, dataType)}) {
      let col = colIn * ${component};
      if (row < uniforms.dimAOuter && col < uniforms.dimBOuter) {
        var value = valueIn;
        let coords = vec3<i32>(batch, row, colIn);
        ${
          hasBias ?
              `value = value + ${isChannelsLast ? 'bias[colIn]' : `${typeSnippet(component, dataType)}(bias[row])`};` :
                                                  ''                                    }
        ${applyActivation}
        ${outputVariable.setByIndices('vec3<u32>(coords)', 'value')}
      }
    }
    `;
      return source;
    };

export const createMatmulProgramInfo =
    (inputs: readonly TensorView[], activationAttributes: InternalActivationAttributes, outputShape: readonly number[],
     reshapedOutputShape?: readonly number[],
     isChannelsLast = false /* only used for conv2dByMatMul*/): ProgramInfo => {
      const aShape = inputs[0].dims;
      const bShape = inputs[1].dims;

      const outerDimsA = aShape.slice(0, -2);
      const outerDimsB = bShape.slice(0, -2);

      const outerDims = reshapedOutputShape ? reshapedOutputShape.slice(0, -2) : outputShape.slice(0, -2);
      const enableBatchUniforms = enableShapesUniforms(outerDims.length);
      const batchShapeOrRank = enableBatchUniforms ? outerDims.length : outerDims;
      const batchDims = inputVariable('batchDims', inputs[0].dataType, batchShapeOrRank, 1, true);
      const variables = [batchDims];
      const batchShapes = [outerDimsA, outerDimsB, outerDims];
      const batchSize = ShapeUtil.size(outerDims);

      const dimAOuter = aShape[aShape.length - 2];
      const dimInner = aShape[aShape.length - 1];
      const dimBOuter = bShape[bShape.length - 1];
      const isVec4 = dimInner % 4 === 0 && dimBOuter % 4 === 0;

      // TODO: fine tune size
      const elementsPerThread = dimAOuter <= 8 ? [4, 1, 1] : [4, 4, 1];
      const workgroupSize: [number, number, number] = [8, 8, 1];
      const dispatch = [
        Math.ceil(dimBOuter / workgroupSize[0] / elementsPerThread[0]),
        Math.ceil(dimAOuter / workgroupSize[1] / elementsPerThread[1]),
        Math.ceil(batchSize / workgroupSize[2] / elementsPerThread[2])
      ];

      const dataType = tensorTypeToWsglStorageType(inputs[0].dataType);
      const components = isVec4 ? 4 : 1;

      const aShapeTemp = [...outerDimsA, dimAOuter, dimInner / components];
      const enableAShapesUniforms = enableShapesUniforms(aShapeTemp.length);
      const aShapeOrRank = enableAShapesUniforms ? aShapeTemp.length : aShapeTemp;

      const bShapeTemp = [...outerDimsB, dimInner, dimBOuter / components];
      const enableBShapesUniforms = enableShapesUniforms(bShapeTemp.length);
      const bShapeOrRank = enableBShapesUniforms ? bShapeTemp.length : bShapeTemp;

      const outputShapeTemp = [batchSize, dimAOuter, dimBOuter / components];
      const enableOutputShapesUniforms = enableShapesUniforms(outputShapeTemp.length);
      const outputShapeOrRank = enableOutputShapesUniforms ? outputShapeTemp.length : outputShapeTemp;

      const A = inputVariable('a', inputs[0].dataType, aShapeOrRank, components);
      const B = inputVariable('b', inputs[1].dataType, bShapeOrRank, components);
      const output = outputVariable('result', inputs[0].dataType, outputShapeOrRank, components);
      variables.push(A);
      variables.push(B);
      variables.push(output);
      const inputVariables = [batchDims, A, B];
      const programUniforms: ProgramUniform[] =
          [{type: 'int32', data: dimAOuter}, {type: 'int32', data: dimBOuter}, {type: 'int32', data: dimInner}];
      if (enableBatchUniforms) {
        programUniforms.push(...createTensorShapeVariables(outerDims));
      }
      if (enableAShapesUniforms) {
        programUniforms.push(...createTensorShapeVariables(aShapeTemp));
      }
      if (enableBShapesUniforms) {
        programUniforms.push(...createTensorShapeVariables(bShapeTemp));
      }
      const inputDependencies: ProgramInputTensorInfoDependency[] = [];
      inputDependencies.push(enableAShapesUniforms ? 'rank' : 'dims');
      inputDependencies.push(enableBShapesUniforms ? 'rank' : 'dims');

      const hasBias = inputs.length > 2;
      const {activationFunction, applyActivation} = getActivationSnippet(activationAttributes, output.type.value);
      const declareFunctions =
          matMulReadWriteFnSource(components, hasBias, applyActivation, variables, batchShapes, isChannelsLast);
      if (hasBias) {
        const enableBiasShapesUniforms = enableShapesUniforms(inputs[2].dims.length);
        const biasShapeOrRank = enableBShapesUniforms ? inputs[2].dims.length : inputs[2].dims;

        const biasComponents = isChannelsLast ? components : 1;
        inputVariables.push(inputVariable('bias', inputs[2].dataType, biasShapeOrRank, biasComponents));
        if (enableBiasShapesUniforms) {
          programUniforms.push(...createTensorShapeVariables(inputs[2].dims));
        }
        inputDependencies.push(enableBiasShapesUniforms ? 'rank' : 'dims');
      }
      if (enableOutputShapesUniforms) {
        programUniforms.push(...createTensorShapeVariables(outputShapeTemp));
      }

      const getShaderSource = (shaderHelper: ShaderHelper) => `
  ${
          shaderHelper.registerUniform('dimAOuter', 'i32')
              .registerUniform('dimBOuter', 'i32')
              .registerUniform('dimInner', 'i32')
              .declareVariables(...inputVariables, output)}
  ${activationFunction}
  ${declareFunctions}
  ${
          isVec4 ? makeMatMulPackedVec4Source(elementsPerThread, workgroupSize, dataType, batchDims) :
                   makeMatMulPackedSource(elementsPerThread, workgroupSize, dataType, batchDims)}
                   `;
      // TODO: turn clipMax and clipMin to uniforms.
      return {
        name: 'MatMul',
        shaderCache: {
          hint: activationAttributes.activationCacheKey + `${elementsPerThread}` +
              `${activationAttributes.activation}` +
              `${activationAttributes.clipMax}` +
              `${activationAttributes.clipMin}` +
              `${isVec4}` +
              `${hasBias}` +
              `${isChannelsLast}`,
          inputDependencies
        },
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
          dispatchGroup: {x: dispatch[0], y: dispatch[1], z: dispatch[2]},
          programUniforms
        }),
        getShaderSource,
      };
    };
