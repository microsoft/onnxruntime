// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {ProgramInfo, ProgramInputTensorInfoDependency, ProgramUniform} from '../types';

import {createTensorShapeVariables, IndicesHelper, inputVariable, internalVariable, outputVariable, ShaderHelper, tensorTypeToWsglStorageType, UniformsArrayType} from './common';
import {MatMulNBitsAttributes} from './matmulnbits';

export const makeMatMulNBitsSource =
    (component: 1|2|3|4 = 1, threadPerCol: 1|4 = 1, workgroupSize: [number, number, number], batchDims: IndicesHelper):
        string => {
          const tileAOuter = workgroupSize[1];
          const tileBOuter = workgroupSize[0];
          const tileInner = workgroupSize[0];
          const tileAWidth = tileInner;
          const tileAHight = tileAOuter;
          const tileBWidth = tileInner;
          const tileBHight = tileBOuter * threadPerCol;

          const matmulSnippet = `
let tileRow = i32(localId.y);
let tileCol = i32(localId.x);
let globalRow = i32(globalId.y);
let globalCol = i32(globalId.x);

// Loop over shared dimension.
for (var t = 0; t < num_tiles; t++) {
  // Load one tile of A into local memory.
  read_a_to_shared_memory(tileRow, tileCol, batch, i32(workgroupId.y) * ${tileAOuter} + tileRow,
      kStart + tileCol${batchDims ? ', batchIndices' : ''});

  // Load one tile of dequantized B into local memory.
  read_dequantized_b_to_shared_memory(${threadPerCol} * tileRow, tileCol,
      (i32(workgroupId.x) * ${tileBOuter} + tileRow) * ${threadPerCol}, kStart + tileCol);

  kStart += tileInner;
  workgroupBarrier();

  // Compute acc values for a single thread.
  for(var j = 0; j < ${threadPerCol}; j++) {
    for (var k = 0; k < tileInner; k = k + 1) {
      acc[j] += dot(mm_a_sub[tileRow][k], mm_dequantized_b_sub[${threadPerCol} * tileCol +j][k]);
    }
  }
  workgroupBarrier();
}

write_result(batch, globalRow, globalCol, acc);
`;

          return `
        var<workgroup> mm_a_sub: array<array<${tensorTypeToWsglStorageType(batchDims.type.tensor, component)}, ${
              tileAWidth}>, ${tileAHight}>;
        var<workgroup> mm_dequantized_b_sub: array<array<${
              tensorTypeToWsglStorageType(batchDims.type.tensor, component)}, ${tileBWidth}>, ${tileBHight}>;

        const tileInner = ${tileInner};

@compute @workgroup_size(${workgroupSize[0]}, ${workgroupSize[1]}, ${workgroupSize[2]})
fn main(@builtin(local_invocation_id) localId : vec3<u32>,
        @builtin(global_invocation_id) globalId : vec3<u32>,
        @builtin(workgroup_id) workgroupId : vec3<u32>) {
    let batch = i32(globalId.z);
    ${batchDims ? `let batchIndices = ${batchDims.offsetToIndices('u32(batch)')};` : ''}
    let num_tiles = (uniforms.dim_inner - 1) / tileInner + 1;
    var kStart = 0;
    var acc = ${tensorTypeToWsglStorageType(batchDims.type.tensor, threadPerCol)}(0);

    ${matmulSnippet}
  }
`;
        };

const matMulNBitsReadWriteFnSource =
    (component: 1|2|3|4 = 1, threadPerCol: 1|4 = 1, variables: IndicesHelper[]): string => {
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
    fn read_a_to_shared_memory(inputRow: i32, inputCol: i32, batch: i32, row: i32, col: i32, batchIndices: ${
          batchVariable.type.indices}) {
      var value = ${tensorTypeToWsglStorageType(aVariable.type.tensor, component)}(0.0);
      if(row < uniforms.dim_a_outer && col < uniforms.dim_inner) {
        ${getAIndices()}
        value = ${aVariable.getByIndices('aIndices')};
      }
      mm_a_sub[inputRow][inputCol] = value;
    }

    fn read_dequantized_b_to_shared_memory(inputRow: i32, inputCol: i32, row: i32, col: i32) {
      for (var i = 0; i < ${threadPerCol}; i++) {
        var value_b = ${tensorTypeToWsglStorageType(bVariable.type.tensor)}(0);
        var value_scale = ${dataType}(0);
        let dequantized_b_col = col / 2;
        let dequantized_scale_col = col / (uniforms.dim_inner /uniforms.n_blocks_per_col);
        if(row + i < i32(uniforms.b_shape[0]) && dequantized_b_col < i32(uniforms.b_shape[1])
            && dequantized_scale_col < i32(uniforms.scales_shape[1])) {
          let bIndices = vec2<u32>(u32(row + i), u32(dequantized_b_col));
          value_b = ${bVariable.getByIndices('bIndices')};
          let scalesIndices = vec2<u32>(u32(row + i), u32(dequantized_scale_col));
          value_scale = ${scalesVariable.getByIndices('scalesIndices')};
        }

        var dequantized_b_value : ${tensorTypeToWsglStorageType(aVariable.type.tensor, component)};
        if (col % 2 == 0) {
          dequantized_b_value = ${tensorTypeToWsglStorageType(aVariable.type.tensor, component)}(
              ${dataType}(extractBits(value_b, 0, 4)), ${dataType}(extractBits(value_b, 4, 4)),
              ${dataType}(extractBits(value_b, 8, 4)), ${dataType}(extractBits(value_b, 12, 4)));
        } else {
          dequantized_b_value = ${tensorTypeToWsglStorageType(aVariable.type.tensor, component)}(
              ${dataType}(extractBits(value_b, 16, 4)), ${dataType}(extractBits(value_b, 20, 4)),
              ${dataType}(extractBits(value_b, 24, 4)), ${dataType}(extractBits(value_b, 28, 4)));
        }

        // The default zero point is 8 for unsigned 4-bit quantization.
        let zero_point = ${tensorTypeToWsglStorageType(batchVariable.type.tensor)}(8);
        mm_dequantized_b_sub[inputRow + i][inputCol] = (dequantized_b_value - zero_point) * value_scale;
      }
    }

    fn write_result(batch: i32, row: i32, col: i32, value: ${
          tensorTypeToWsglStorageType(outputVariable.type.tensor, threadPerCol)}) {
      if (row < uniforms.dim_a_outer && col < uniforms.dim_b_outer) {
        let coords = vec3<i32>(batch, row, col);
        ${outputVariable.setByIndices('vec3<u32>(coords)', 'value')}
      }
    }
    `;
      return source;
    };

export const createMatMulNBitsSharedProgramInfo =
    (inputs: readonly TensorView[], attributes: MatMulNBitsAttributes, outputShape: readonly number[]): ProgramInfo => {
      const component = 4;     // Only support component = 4
      const threadPerCol = 4;  // only support thread = 4 per col
      const aShape = inputs[0].dims;
      const bShape = inputs[1].dims;
      const outerDimsA = aShape.slice(0, -2);
      const outerDims = outputShape.slice(0, -2);
      const batchSize = ShapeUtil.size(outerDims);
      const dimAOuter = aShape[aShape.length - 2];
      const dimInner = aShape[aShape.length - 1];
      const dimBOuter = bShape[0];

      // TODO: fine tune size
      const workgroupSize: [number, number, number] = [8, 8, 1];
      const dispatch = [
        Math.ceil(dimBOuter / workgroupSize[0] / threadPerCol), Math.ceil(dimAOuter / workgroupSize[1]),
        Math.ceil(batchSize / workgroupSize[2])
      ];

      const aShapeTemp = [...outerDimsA, dimAOuter, dimInner / component];
      const aRank = aShapeTemp.length;
      const bShapeTemp = [dimBOuter, dimInner / 8];  // 8 = bitsof(uint32) / attributes.bits
      const bRank = bShapeTemp.length;
      const outputShapeTemp = [batchSize, dimAOuter, dimBOuter / threadPerCol];
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

        const A = inputVariable('a', inputs[0].dataType, aRank, 4);
        const B = inputVariable('b', DataType.uint32, bRank, 1);
        const Scales = inputVariable('scales', inputs[0].dataType, scalesRank, 1);
        const output = outputVariable('result', inputs[0].dataType, outputShapeTemp.length, 4);
        const inputVariables = [A, B, Scales];

        const uniforms: UniformsArrayType = [
          {name: 'dim_a_outer', type: 'i32'}, {name: 'dim_b_outer', type: 'i32'}, {name: 'dim_inner', type: 'i32'},
          {name: 'n_blocks_per_col', type: 'i32'}
        ];
        const declareFunctions =
            matMulNBitsReadWriteFnSource(component, threadPerCol, [batchDims, A, B, Scales, output]);
        return `
  ${
            shaderHelper.registerUniforms(uniforms).registerInternalVariables(batchDims).declareVariables(
                ...inputVariables, output)}
  ${declareFunctions}
  ${makeMatMulNBitsSource(component, threadPerCol, workgroupSize, batchDims)}
                   `;
      };
      return {
        name: 'MatMulNBits',
        shaderCache: {inputDependencies},
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
          dispatchGroup: {x: dispatch[0], y: dispatch[1], z: dispatch[2]},
          programUniforms
        }),
        getShaderSource,
      };
    };
