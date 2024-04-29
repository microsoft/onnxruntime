// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {ProgramInfo, ProgramInputTensorInfoDependency, ProgramUniform} from '../types';

import {createTensorShapeVariables, IndicesHelper, inputVariable, internalVariable, outputVariable, ShaderHelper, tensorTypeToWsglStorageType, UniformsArrayType} from './common';
import {MatMulNBitsAttributes} from './matmulnbits';

export const makeMatMulNBitsSpecialASource =
    (component: 1|2|3|4 = 1, compenentsPerCol: 1|4 = 1, colsPerThread: 1|4 = 1, workgroupSize: [number, number, number],
     batchDims: IndicesHelper): string => {
      const tileAOuter = 1;
      const tileInner = workgroupSize[0];
      const tileAWidth = 2 * tileInner;
      const tileAHight = tileAOuter;
      const tileResWidth = 2 * tileInner;
      const tileResHight = compenentsPerCol * colsPerThread;

      const matmulSnippet = `
let tileRow = i32(localId.y);
let tileCol = i32(localId.x);
let globalRow = i32(globalId.y);
let globalCol = i32(globalId.x);
let workgroupIdX = i32(workgroupId.x);
mm_res_sub[0][tileCol] = ${tensorTypeToWsglStorageType(batchDims.type.tensor)}(0);
mm_res_sub[1][tileCol] = ${tensorTypeToWsglStorageType(batchDims.type.tensor)}(0);
mm_res_sub[2][tileCol] = ${tensorTypeToWsglStorageType(batchDims.type.tensor)}(0);
mm_res_sub[3][tileCol] = ${tensorTypeToWsglStorageType(batchDims.type.tensor)}(0);

mm_res_sub[0][tileCol + 32] = ${tensorTypeToWsglStorageType(batchDims.type.tensor)}(0);
mm_res_sub[1][tileCol + 32] = ${tensorTypeToWsglStorageType(batchDims.type.tensor)}(0);
mm_res_sub[2][tileCol + 32] = ${tensorTypeToWsglStorageType(batchDims.type.tensor)}(0);
mm_res_sub[3][tileCol + 32] = ${tensorTypeToWsglStorageType(batchDims.type.tensor)}(0);

// Loop over shared dimension.
var acc = ${tensorTypeToWsglStorageType(batchDims.type.tensor)}(0);
var dequantized_b = mat2x4<${tensorTypeToWsglStorageType(batchDims.type.tensor)}>(${
          tensorTypeToWsglStorageType(
              batchDims.type.tensor,
              component)}(0), ${tensorTypeToWsglStorageType(batchDims.type.tensor, component)}(0));
for (var t = 0; t < num_tiles; t++) {
  // Load one tile of A into local memory.
  read_a_to_shared_memory(0, tileCol, batch, 0,
      kStartA + tileCol${batchDims ? ', batchIndices' : ''});

  workgroupBarrier();

  dequantized_b = read_dequantized_b_to_shared_memory(workgroupIdX * 4, kStartB + tileCol);
  acc = dot(mm_a_sub[0][2 * tileCol], dequantized_b[0]);
  mm_res_sub[0][2 * tileCol] += acc;
  acc = dot(mm_a_sub[0][2 * tileCol + 1], dequantized_b[1]);
  mm_res_sub[0][2 * tileCol + 1] += acc;

  dequantized_b = read_dequantized_b_to_shared_memory(workgroupIdX * 4 + 1, kStartB + tileCol);
  acc = dot(mm_a_sub[0][2 * tileCol], dequantized_b[0]);
  mm_res_sub[1][2 * tileCol] += acc;
  acc = dot(mm_a_sub[0][2 * tileCol + 1], dequantized_b[1]);
  mm_res_sub[1][2 * tileCol + 1] += acc;

  dequantized_b = read_dequantized_b_to_shared_memory(workgroupIdX * 4 + 2, kStartB + tileCol);
  acc = dot(mm_a_sub[0][2 * tileCol], dequantized_b[0]);
  mm_res_sub[2][2 * tileCol] += acc;
  acc = dot(mm_a_sub[0][2 * tileCol + 1], dequantized_b[1]);
  mm_res_sub[2][2 * tileCol + 1] += acc;

  dequantized_b = read_dequantized_b_to_shared_memory(workgroupIdX * 4 + 3, kStartB + tileCol);
  acc = dot(mm_a_sub[0][2 * tileCol], dequantized_b[0]);
  mm_res_sub[3][2 * tileCol] += acc;
  acc = dot(mm_a_sub[0][2 * tileCol + 1], dequantized_b[1]);
  mm_res_sub[3][2 * tileCol + 1] += acc;

  kStartA += tileInner;
  kStartB += tileInnerB ;
  workgroupBarrier();
}

mm_res_sub[0][tileCol] += mm_res_sub[0][tileCol + 32];
mm_res_sub[1][tileCol] += mm_res_sub[1][tileCol + 32];
mm_res_sub[2][tileCol] += mm_res_sub[2][tileCol + 32];
mm_res_sub[3][tileCol] += mm_res_sub[3][tileCol + 32];

workgroupBarrier();

if (tileCol < 16) {
  mm_res_sub[0][tileCol] += mm_res_sub[0][tileCol + 16];
  mm_res_sub[1][tileCol] += mm_res_sub[1][tileCol + 16];
} else {
  mm_res_sub[2][tileCol - 16] += mm_res_sub[2][tileCol];
  mm_res_sub[3][tileCol - 16] += mm_res_sub[3][tileCol];
}

workgroupBarrier();

if (tileCol < 4) {
  mm_res_sub[tileCol][0] += mm_res_sub[tileCol][1] +
  mm_res_sub[tileCol][2] + mm_res_sub[tileCol][3] +
  mm_res_sub[tileCol][4] + mm_res_sub[tileCol][5] +
  mm_res_sub[tileCol][6] + mm_res_sub[tileCol][7] +
  mm_res_sub[tileCol][8] + mm_res_sub[tileCol][9] +
  mm_res_sub[tileCol][10] + mm_res_sub[tileCol][11] +
  mm_res_sub[tileCol][12] + mm_res_sub[tileCol][13] +
  mm_res_sub[tileCol][14] + mm_res_sub[tileCol][15];
}

workgroupBarrier();

if (tileCol == 0) {
  write_result(batch, 0, workgroupIdX, ${
          tensorTypeToWsglStorageType(batchDims.type.tensor, component)}(mm_res_sub[0][0],
    mm_res_sub[1][0], mm_res_sub[2][0], mm_res_sub[3][0]));
}
`;

      return `
        var<workgroup> mm_a_sub: array<array<${tensorTypeToWsglStorageType(batchDims.type.tensor, component)}, ${
          tileAWidth}>, ${tileAHight}>;
        var<workgroup> mm_res_sub: array<array<${tensorTypeToWsglStorageType(batchDims.type.tensor)}, ${
          tileResWidth} + 1>, ${tileResHight}>;

        const tileInner = ${tileInner * 2};
        const tileInnerB = ${tileInner};

@compute @workgroup_size(${workgroupSize[0]}, ${workgroupSize[1]}, ${workgroupSize[2]})
fn main(@builtin(local_invocation_id) localId : vec3<u32>,
        @builtin(global_invocation_id) globalId : vec3<u32>,
        @builtin(workgroup_id) workgroupId : vec3<u32>) {
    let batch = i32(globalId.z);
    ${batchDims ? `let batchIndices = ${batchDims.offsetToIndices('u32(batch)')};` : ''}
    let num_tiles = (uniforms.dim_inner - 1) / tileInner + 1;
    var kStartA = 0;
    var kStartB = 0;

    ${matmulSnippet}
  }
`;
    };

const matMulNBitsReadWriteFnSource =
    (component: 1|2|3|4 = 1, compenentsPerCol: 1|4 = 1, variables: IndicesHelper[]): string => {
      const [batchVariable, aVariable, bVariable, scalesVariable, outputVariable] = variables;
      const dataType = tensorTypeToWsglStorageType(variables[0].type.tensor);
      const getAIndices = () => {
        const aRank = aVariable.rank;
        const batchRank = batchVariable.rank;
        let resStr = `var aIndices: ${aVariable.type.indices};`;
        for (let i = aRank - 2 - 1, j = batchRank - 1; i >= 0; i--, j--) {
          resStr += `\naIndices[${i}] = ${batchRank > 1 ? `batchIndices[${j}]` : 'batchIndices'};`;
        }

        resStr += `\naIndices[${aRank - 2}] = u32(row);
                   aIndices[${aRank - 1}] = u32(col);`;
        return resStr;
      };
      const source = `
    fn read_a_to_shared_memory(inputRow: i32, inputCol: i32, batch: i32, row: i32, colIn: i32, batchIndices: ${
          batchVariable.type.indices}) {
      var col = colIn;
      var value = ${tensorTypeToWsglStorageType(aVariable.type.tensor, component)}(0.0);
      if(row < uniforms.dim_a_outer && col < uniforms.dim_inner) {
        ${getAIndices()}
        value = ${aVariable.getByIndices('aIndices')};
      }
      mm_a_sub[inputRow][inputCol] = value;
      value = ${tensorTypeToWsglStorageType(aVariable.type.tensor, component)}(0.0);
      col = colIn + 32;
      if(row < uniforms.dim_a_outer && col < uniforms.dim_inner) {
        ${getAIndices()}
        value = ${aVariable.getByIndices('aIndices')};
      }
      mm_a_sub[inputRow][inputCol + 32] = value;
    }

    fn read_dequantized_b_to_shared_memory(row: i32, col: i32) -> mat2x4<${
          tensorTypeToWsglStorageType(batchVariable.type.tensor)}> {
        var value_b = ${tensorTypeToWsglStorageType(bVariable.type.tensor)}(0);
        var value_scale_0 = ${dataType}(0);
        var value_scale_1 = ${dataType}(0);
        let dequantized_b_col = col;
        let dequantized_scale_col_0 = 2 * col / (uniforms.dim_inner /uniforms.n_blocks_per_col);
        let dequantized_scale_col_1 = (2 * col + 1) / (uniforms.dim_inner /uniforms.n_blocks_per_col);
        if(row < i32(uniforms.b_shape[0]) && dequantized_b_col < i32(uniforms.b_shape[1])
            && dequantized_scale_col_1 < i32(uniforms.scales_shape[1])) {
          let bIndices = vec2<u32>(u32(row), u32(dequantized_b_col));
          value_b = ${bVariable.getByIndices('bIndices')};
          var scalesIndices = vec2<u32>(u32(row), u32(dequantized_scale_col_0));
          value_scale_0 = ${scalesVariable.getByIndices('scalesIndices')};
          scalesIndices = vec2<u32>(u32(row), u32(dequantized_scale_col_1));
          value_scale_1 = ${scalesVariable.getByIndices('scalesIndices')};
        }

        let dequantized_b_value_0 = ${tensorTypeToWsglStorageType(aVariable.type.tensor, component)}(
            ${dataType}(extractBits(value_b, 0, 4)), ${dataType}(extractBits(value_b, 4, 4)),
            ${dataType}(extractBits(value_b, 8, 4)), ${dataType}(extractBits(value_b, 12, 4)));
        let dequantized_b_value_1 = ${tensorTypeToWsglStorageType(aVariable.type.tensor, component)}(
            ${dataType}(extractBits(value_b, 16, 4)), ${dataType}(extractBits(value_b, 20, 4)),
            ${dataType}(extractBits(value_b, 24, 4)), ${dataType}(extractBits(value_b, 28, 4)));

        // The default zero point is 8 for unsigned 4-bit quantization.
        let zero_point = ${tensorTypeToWsglStorageType(batchVariable.type.tensor)}(8);
        return mat2x4((dequantized_b_value_0 - zero_point) * value_scale_0, (dequantized_b_value_1 - zero_point) * value_scale_1);
    }

    fn write_result(batch: i32, row: i32, col: i32, value: ${
          tensorTypeToWsglStorageType(outputVariable.type.tensor, compenentsPerCol)}) {
      if (row < uniforms.dim_a_outer && col < uniforms.dim_b_outer) {
        let coords = vec3<i32>(batch, row, col);
        ${outputVariable.setByIndices('vec3<u32>(coords)', 'value')}
      }
    }
    `;
      return source;
    };

export const createMatMulNBitsSpecialAProgramInfo =
    (inputs: readonly TensorView[], attributes: MatMulNBitsAttributes, outputShape: readonly number[]): ProgramInfo => {
      const component = 4;  // Only support component = 4
      const componentsPerCol = 4;
      const colsPerThread = 1;
      const aShape = inputs[0].dims;
      const bShape = inputs[1].dims;
      const outerDims = outputShape.slice(0, -2);
      const batchSize = ShapeUtil.size(outerDims);
      const dimAOuter = aShape[aShape.length - 2];
      const dimInner = aShape[aShape.length - 1];
      const dimBOuter = bShape[0];

      // TODO: fine tune size
      const workgroupSize: [number, number, number] = [32, 1, 1];
      const dispatch = [
        Math.ceil(dimBOuter / (componentsPerCol * colsPerThread)), Math.ceil(dimAOuter / workgroupSize[1]),
        Math.ceil(batchSize / workgroupSize[2])
      ];

      const aShapeTemp = [1, 1, dimInner / component];
      const aRank = aShapeTemp.length;
      const bShapeTemp = [dimBOuter, dimInner / 8];  // 8 = bitsof(uint32) / attributes.bits
      const bRank = bShapeTemp.length;
      const outputShapeTemp = [1, 1, dimBOuter];
      const nBlocksPerCol = Math.floor((attributes.k + attributes.blockSize - 1) / attributes.blockSize);
      const programUniforms: ProgramUniform[] = [
        {type: DataType.int32, data: dimAOuter}, {type: DataType.int32, data: dimBOuter},
        {type: DataType.int32, data: dimInner / component}, {type: DataType.int32, data: nBlocksPerCol}
      ];
      const scaleShapeTemp = [attributes.n, nBlocksPerCol];
      const scalesRank = bShapeTemp.length;
      const inputDependencies: ProgramInputTensorInfoDependency[] = ['rank', 'rank', 'rank'];
      programUniforms.push(...createTensorShapeVariables(outerDims, aShapeTemp, bShapeTemp, scaleShapeTemp));


      programUniforms.push(...createTensorShapeVariables(outputShapeTemp));

      const getShaderSource = (shaderHelper: ShaderHelper) => {
        const batchRank = outerDims.length;
        const batchDims = internalVariable('batchDims', inputs[0].dataType, batchRank, 1);

        const A = inputVariable('a', inputs[0].dataType, aRank, component);
        const B = inputVariable('b', DataType.uint32, bRank, 1);
        const Scales = inputVariable('scales', inputs[0].dataType, scalesRank, 1);
        const output = outputVariable('result', inputs[0].dataType, outputShapeTemp.length, 4);
        const inputVariables = [A, B, Scales];

        const uniforms: UniformsArrayType = [
          {name: 'dim_a_outer', type: 'i32'}, {name: 'dim_b_outer', type: 'i32'}, {name: 'dim_inner', type: 'i32'},
          {name: 'n_blocks_per_col', type: 'i32'}
        ];
        const declareFunctions =
            matMulNBitsReadWriteFnSource(component, componentsPerCol, [batchDims, A, B, Scales, output]);
        return `
  ${
            shaderHelper.registerUniforms(uniforms).registerInternalVariables(batchDims).declareVariables(
                ...inputVariables, output)}
  ${declareFunctions}
  ${makeMatMulNBitsSpecialASource(component, componentsPerCol, colsPerThread, workgroupSize, batchDims)}
                   `;
      };
      return {
        name: 'MatMulNBitsSpecialA',
        shaderCache: {inputDependencies},
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
          dispatchGroup: {x: dispatch[0], y: dispatch[1], z: dispatch[2]},
          programUniforms
        }),
        getShaderSource,
      };
    };
