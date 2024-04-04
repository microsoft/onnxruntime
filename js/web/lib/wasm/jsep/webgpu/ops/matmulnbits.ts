// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType, getTensorElementSize} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo, ProgramUniform} from '../types';

import {createTensorShapeVariables, getMaxComponents, inputVariable, outputVariable, ShaderHelper, tensorTypeToWsglStorageType, UniformsArrayType} from './common';

//  TODO support quantization bits not equal to 4
export interface MatMulNBitsAttributes extends AttributeWithCacheKey {
  k: number;
  n: number;
  accuracyLevel: number;
  bits: number;
  blockSize: number;
}

const validateInputs = (inputs: readonly TensorView[], attributes: MatMulNBitsAttributes): void => {
  if (inputs.length < 3 || inputs.length > 4) {
    throw new Error('MatMulNBits requires 3 or 4 inputs');
  }
  const a = inputs[0];
  const aRank = a.dims.length;
  if (a.dims[aRank - 1] !== attributes.k) {
    throw new Error('The last dim of input shape does not match the k value');
  }
  const nBlocksPerCol = Math.floor((attributes.k + attributes.blockSize - 1) / attributes.blockSize);
  const blobSize = attributes.blockSize / 8 * attributes.bits;
  const b = inputs[1];
  if (!ShapeUtil.areEqual(b.dims, [attributes.n, nBlocksPerCol, blobSize])) {
    throw new Error('The second inputs must be 3D tensor with shape N X nBlocksPerCol X blobSize');
  }
  const scales = inputs[2];
  const scalesShape = scales.dims;
  if (ShapeUtil.size(scalesShape) !== attributes.n * nBlocksPerCol) {
    throw new Error('scales input size error.');
  }
  if (inputs.length === 4) {
    const zeroPoints = inputs[3];
    const zeroPointsShape = zeroPoints.dims;
    const expectedZeroPointsSize =
        attributes.bits > 4 ? (attributes.n * nBlocksPerCol) : attributes.n * Math.floor((nBlocksPerCol + 1) / 2);
    if (ShapeUtil.size(zeroPointsShape) !== expectedZeroPointsSize) {
      throw new Error('zeroPoints input size error.');
    }
  }
};

export const createBlockwiseMatMulNBitsProgramInfo =
    (inputs: readonly TensorView[], attributes: MatMulNBitsAttributes,
     maxComputeWorkgroupSizes: [number, number, number], maxComputeWorkgroupStorageSize: number): ProgramInfo => {
      const inputShape = inputs[0].dims;
      const aRank = inputShape.length;
      const nBlocksPerCol = Math.floor((attributes.k + attributes.blockSize - 1) / attributes.blockSize);
      const dimAOuter = inputShape[aRank - 2];
      const dimInner = attributes.k;
      const dimBOuter = attributes.n;
      const batchDims = inputShape.slice(0, aRank - 2);
      const batchSize = ShapeUtil.size(batchDims);
      const blobSize = attributes.blockSize / 8 * attributes.bits;
      const blobSizeInWords = blobSize / 4;
      const outputShape = batchDims.concat([dimAOuter, dimBOuter]);
      const components = getMaxComponents(dimBOuter);
      const aComponents = getMaxComponents(attributes.k);
      const bComponents = getMaxComponents(blobSizeInWords);
      const elementSize = getTensorElementSize(inputs[0].dataType);
      if (!elementSize) {
        throw new Error(`Unsupported data type: ${inputs[0].dataType}`);
      }
      const requiredStorageSizePerWorkgroupX = dimAOuter * elementSize * components;
      // TODO use alternative implementation if requiredStorageSizePerWorkgroupX is too large
      if (requiredStorageSizePerWorkgroupX > maxComputeWorkgroupStorageSize) {
        throw new Error('The required storage size per workgroup is too large.');
      }
      const maxWorkgroupsizeX = Math.ceil(maxComputeWorkgroupStorageSize / requiredStorageSizePerWorkgroupX);
      const maxWorkgroupSizeX = Math.min(maxComputeWorkgroupSizes[0], nBlocksPerCol, maxWorkgroupsizeX);
      // Find the largest workgroupSizeX that divides nBlocksPerCol.
      const workgroupSizeX = (() => {
        for (let i = maxWorkgroupSizeX; i > 1; i--) {
          if (nBlocksPerCol % i === 0) {
            return i;
          }
        }
        return 1;
      })();
      const workgroupSize = [workgroupSizeX, 1, 1];
      const dispatch = [Math.ceil(dimAOuter / workgroupSize[0]), Math.ceil(dimBOuter / components), batchSize];

      const programUniforms: ProgramUniform[] = [{type: DataType.uint32, data: attributes.blockSize}];
      const inputShapeTemp = [batchSize, dimAOuter, dimInner / aComponents];
      const bShape = ShapeUtil.convertShape(inputs[1].dims).slice();
      bShape.splice(-1, 1, blobSizeInWords / bComponents);
      programUniforms.push(...createTensorShapeVariables(inputShapeTemp));
      programUniforms.push(...createTensorShapeVariables(bShape));
      programUniforms.push(...createTensorShapeVariables(inputs[2].dims));
      if (inputs.length === 4) {
        programUniforms.push(...createTensorShapeVariables(ShapeUtil.convertShape(inputs[3].dims)));
      }
      const outputShapeTemp = [batchSize, dimAOuter, dimBOuter / components];
      programUniforms.push(...createTensorShapeVariables(outputShapeTemp));
      const getShaderSource = (shaderHelper: ShaderHelper) => {
        const inputRank = inputShapeTemp.length;
        const a = inputVariable('a', inputs[0].dataType, inputRank, aComponents);
        const b = inputVariable('b', DataType.uint32, bShape.length, bComponents);
        const scales = inputVariable('scales', inputs[2].dataType, inputs[2].dims.length);
        const inputVariables = [a, b, scales];
        const zeroPoints =
            inputs.length === 4 ? inputVariable('zero_points', DataType.uint32, inputs[3].dims.length) : undefined;
        if (zeroPoints) {
          inputVariables.push(zeroPoints);
        }
        const outputRank = outputShapeTemp.length;
        const output = outputVariable('output', inputs[0].dataType, outputRank, components);
        const uniforms: UniformsArrayType = [{name: 'block_size', type: 'u32'}];
        const dataType = tensorTypeToWsglStorageType(inputs[0].dataType);

        const qDqDataType = (() => {
          switch (aComponents) {
            case 1:
              return `array<${dataType}, 8>`;
            case 2:
              return `mat4x2<${dataType}>`;
            case 4:
              return `mat2x4<${dataType}>`;
            default:
              throw new Error(`${aComponents}-component is not supported.`);
          }
        })();

        const zeroPointsBytesPerCol = Math.floor((nBlocksPerCol + 1) / 2);
        return `
        const block_size = ${attributes.blockSize};
        var<workgroup> workgroup_shared: array<${output.type.value}, ${dimAOuter * workgroupSize[0]}>;
        ${shaderHelper.registerUniforms(uniforms).declareVariables(...inputVariables, output)}
        ${shaderHelper.mainStart([
          workgroupSize[0], workgroupSize[1], workgroupSize[2]
        ])}
          var output_indices: ${output.type.indices};
          var a_indices: ${a.type.indices};
          var block = local_id.x;
          var col = workgroup_id.y;
          var batch = workgroup_id.z;
          var a_data_array: array<array<${a.type.value}, block_size / ${aComponents}>, ${dimAOuter}>;
          ${a.indicesSet('a_indices', '0', 'batch')};
          for (var m: u32 = 0; m < ${dimAOuter}; m++) {
            ${a.indicesSet('a_indices', '1', 'm')};
            for (var i: u32 = 0; i < block_size / ${aComponents}; i++) {
              ${a.indicesSet('a_indices', '2', `(block_size / ${aComponents}) * block + i`)};
              a_data_array[m][i] = ${a.getByIndices('a_indices')};
            }
          }
          workgroupBarrier();
          // Two zero points are packed into one byte when uniforms.bits is 4.
          for (var c: u32 = 0; c < ${components}; c++) {
            let col_times_components_plus_c = col * ${components} + c;
              ${
            zeroPoints ? `
            var zero_point_byte_count: u32 = col_times_components_plus_c * ${zeroPointsBytesPerCol} + (block >> 0x1u);
            var zero_point_word_index: u32 = zero_point_byte_count >> 0x2u;
            var zero_point_byte_offset: u32 = zero_point_byte_count & 0x3u;
            var zero_point_nibble_offset: u32 = block & 0x1u;
            var zero_point_bits_offset: u32 = (zero_point_byte_offset << 3) + (zero_point_nibble_offset << 2);
            var zero_point_word: u32 = ${zeroPoints.getByOffset('zero_point_word_index')} >> zero_point_bits_offset;` :
                         ''}
            var b_indices: ${b.type.indices};
            ${b.indicesSet('b_indices', '0', 'col_times_components_plus_c')};
            // The scale and zero points are computed per block.
            var scales_index = col_times_components_plus_c * ${nBlocksPerCol} + block;
            let scale = ${scales.getByOffset('scales_index')};
            // The default zero point is 8 for unsigned 4-bit quantization.
            let zero_point = ${dataType}(${zeroPoints ? '(zero_point_word) & 0xFu' : 8.0});
            ${b.indicesSet('b_indices', '1', 'block')};
            var word_offset: u32 = 0;
            for (var word: u32 = 0; word < ${blobSizeInWords}; word += ${bComponents}) {
              ${b.indicesSet('b_indices', '2', 'word')};
              let b_data = ${b.getByIndices('b_indices')};
              for (var i: u32 = 0; i < ${bComponents}; i++) {
                let b_value = ${bComponents === 1 ? 'b_data' : 'b_data[word + i]'};
                let b_quantized_values = ${qDqDataType}(${
            Array.from({length: 8}, (_, i) => `${dataType}((b_value >> ${(i * 4).toString()}) & 0xFu)`).join(', ')});
                let b_dequantized_values = ${(() => {
          if (aComponents === 1) {
            return `return ${qDqDataType}(${
                Array.from({length: 8}, (_, i) => `(b_quantized_values[${i}] - zero_point) * scale`).join(', ')});`;
          } else {
            return `(b_quantized_values - ${qDqDataType}(${Array(8).fill('zero_point').join(',')})) * scale;`;
          }
        })()};
                // Number of B elements per 32-bit word is 32/bits = 32/4 = 8
                var offset: u32 = word_offset;
                ${(() => {
          const code = [];
          for (let j = 0; j < 8; j += aComponents) {
            for (let m = 0; m < dimAOuter; m++) {
              code.push(`workgroup_shared[(block * ${dimAOuter} + ${m})] ${components > 1 ? '[c]' : ''} +=
                            ${
                  aComponents === 1 ? `a_data_array[${m}][offset] * b_dequantized_values[${j}]` :
                                      `dot(a_data_array[${m}][offset], b_dequantized_values[${j} / ${aComponents}]);`};
                        `);
            }
            code.push('offset++;');
          }
          code.push(`word_offset += ${8 / aComponents};`);
          return code.join('\n');
        })()};
              }
            }
          }

          workgroupBarrier();
          if (local_id.x == 0u) {
            ${output.indicesSet('output_indices', '0', 'batch')};
            ${output.indicesSet('output_indices', outputRank - 1, 'col')};
                ${(() => {
          const code = [];
          for (let n = 0; n < dimBOuter; n++) {
            const rhs = Array.from({length: workgroupSize[0]}, (_, b) => `workgroup_shared[${(b * dimAOuter + n)}]`);
            code.push(`let output_value_${n} = ${rhs.join(' + ')};`);
            code.push(`${output.indicesSet('output_indices', outputRank - 2, n)};`);
            code.push(`${output.setByIndices('output_indices', `output_value_${n}`)};`);
          }
          return code.join('\n');
        })()};
          }
        }`;
      };
      return {
        name: 'BlockwiseMatMulNBits',
        shaderCache:
            {hint: `${attributes.cacheKey};${inputs.length}}`, inputDependencies: Array(inputs.length).fill('rank')},
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
          dispatchGroup: {x: dispatch[0], y: dispatch[1], z: dispatch[2]},
          programUniforms
        }),
        getShaderSource
      };
    };

export const matMulNBits = (context: ComputeContext, attributes: MatMulNBitsAttributes): void => {
  validateInputs(context.inputs, attributes);
  const maxComputeWorkgroupSizes: [number, number, number] = context.getMaxComputeWorkgroupSizes();
  const maxComputeWorkgroupStorageSize = context.getMaxComputeWorkgroupStoragesize();
  context.compute(createBlockwiseMatMulNBitsProgramInfo(
      context.inputs, attributes, maxComputeWorkgroupSizes, maxComputeWorkgroupStorageSize));
};

export const parseMatMulNBitsAttributes = (attributes: Record<string, unknown>): MatMulNBitsAttributes =>
    createAttributeWithCacheKey(attributes as Omit<MatMulNBitsAttributes, keyof AttributeWithCacheKey>);
