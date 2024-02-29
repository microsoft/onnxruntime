// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo, ProgramUniform} from '../types';

import {createTensorShapeVariables, inputVariable, outputVariable, ShaderHelper, tensorTypeToWsglStorageType, UniformsArrayType} from './common';

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

export const createMatMulNBitsProgramInfo =
    (inputs: readonly TensorView[], attributes: MatMulNBitsAttributes): ProgramInfo => {
      const a = inputs[0];
      const b = inputs[1];
      const scales = inputs[2];
      const aRank = a.dims.length;
      const outputShape = a.dims.slice(0, aRank - 1).concat(attributes.n);
      const outputSize = ShapeUtil.size(outputShape);


      const programUniforms: ProgramUniform[] = [
        {type: DataType.uint32, data: outputSize}, {type: DataType.uint32, data: attributes.k},
        {type: DataType.uint32, data: attributes.n}, {type: DataType.uint32, data: attributes.accuracyLevel},
        {type: DataType.uint32, data: attributes.bits}, {type: DataType.uint32, data: attributes.blockSize}
      ];
      programUniforms.push(...createTensorShapeVariables(a.dims));
      programUniforms.push(...createTensorShapeVariables(ShapeUtil.convertShape(b.dims)));
      programUniforms.push(...createTensorShapeVariables(scales.dims));
      if (inputs.length === 4) {
        programUniforms.push(...createTensorShapeVariables(ShapeUtil.convertShape(inputs[3].dims)));
      }
      programUniforms.push(...createTensorShapeVariables(outputShape));
      const getShaderSource = (shaderHelper: ShaderHelper) => {
        const a = inputVariable('a', inputs[0].dataType, inputs[0].dims.length);
        const b = inputVariable('b', DataType.uint32, inputs[1].dims.length);
        const scales = inputVariable('scales', inputs[2].dataType, inputs[2].dims.length);
        const inputVariables = [a, b, scales];
        const zeroPoints =
            inputs.length === 4 ? inputVariable('zero_points', DataType.uint32, inputs[3].dims.length) : undefined;
        if (zeroPoints) {
          inputVariables.push(zeroPoints);
        }
        const output = outputVariable('output', inputs[0].dataType, outputShape.length);
        const uniforms: UniformsArrayType = [
          {name: 'output_size', type: 'u32'}, {name: 'k', type: 'u32'}, {name: 'n', type: 'u32'},
          {name: 'accuracy_level', type: 'u32'}, {name: 'bits', type: 'u32'}, {name: 'block_size', type: 'u32'}
        ];
        const nBlocksPerCol = Math.floor((attributes.k + attributes.blockSize - 1) / attributes.blockSize);
        const blobSize = attributes.blockSize / 8 * attributes.bits;
        const wordPerBlob = blobSize / 4;
        const dataType = tensorTypeToWsglStorageType(inputs[0].dataType);
        return `
        fn ortUnpack8x4snorm(value: u32) -> array<${dataType}, 8>{
          var result = array<${dataType}, 8>();
          var offset: u32 = 0;
          let count: u32 = 4;
          for (var i: u32 = 0; i < 8u; i++) {
            result[i] = ${dataType}(extractBits(value, offset, count));
            offset += count;
          }
          return result;
        }
        ${shaderHelper.registerUniforms(uniforms).declareVariables(...inputVariables, output)}
        ${shaderHelper.mainStart()}
          ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.output_size')}
          var value: ${dataType} = 0.0;
          let output_indices = ${output.offsetToIndices('global_idx')};
          var a_indices: ${a.type.indices} = output_indices;
          var n = ${output.indicesGet('output_indices', aRank - 1)};
          // Two zero points are packed into one byte because uniforms.bits <= 4.
          // zero_point_offset is either 0 or 4. It is bit offset within one byte.
          // TODO support zero_point_offset for bits > 4
          ${
            zeroPoints ? `
            var zero_point_index: u32 = n * ((${nBlocksPerCol} + 1) / 2) / 4;
            var zero_point_word: u32 = ${zeroPoints.getByOffset('zero_point_index')};
            var zero_point_offset: u32 = 0;` :
                         ''}
          var scale_idex = n * ${nBlocksPerCol};
          var b_indices: ${b.type.indices};
          ${b.indicesSet('b_indices', '0', 'n')};
          var block_offset: u32 = 0;
          for (var block: u32 = 0; block < ${nBlocksPerCol}; block++) {
            // The scale and zero points are computed per block.
            let scale = ${scales.getByOffset('scale_idex')};
            // The default zero point is 8 for unsigned 4-bit quantization.
            let zero_point: ${dataType} = ${
            zeroPoints ? `${dataType}(extractBits(zero_point_word, zero_point_offset, 4))` : 8.0};
            ${b.indicesSet('b_indices', '1', 'block')};
            var word_offset: u32 = block_offset;
            for (var word: u32 = 0; word < ${wordPerBlob}; word++) {
              ${b.indicesSet('b_indices', '2', 'word')};
              let b_value = ${b.getByIndices('b_indices')};
              let b_quantized_values: array<${dataType}, 8> = ortUnpack8x4snorm(b_value);
              // Number of B elements per 32-bit word is 32/bits = 32/4 = 8
              var offset: u32 = word_offset;
              for (var i: u32 = 0; i < 8; i++) {
                ${a.indicesSet('a_indices', aRank - 1, 'offset')};
                let a_value = ${a.getByIndices('a_indices')};
                let b_quantized_value = b_quantized_values[i];
                let b_dequantized_value = (b_quantized_value - zero_point) * scale;
                value += a_value * b_dequantized_value;
                offset++;
              }
              word_offset += 8;
            }
            scale_idex++;
            ${
            zeroPoints ? `
            if (zero_point_offset == 28) {
              zero_point_offset = 0;
              zero_point_index++;
              zero_point_word = ${zeroPoints.getByOffset('zero_point_index')};
            } else {
              zero_point_offset += 4;
            }` :
                         ''}
            block_offset += uniforms.block_size;
          }
          ${output.setByOffset('global_idx', 'value')};
        }
        `;
      };
      return {
        name: 'MatMulNBits',
        shaderCache:
            {hint: `${attributes.cacheKey};${inputs.length}`, inputDependencies: Array(inputs.length).fill('rank')},
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
          dispatchGroup: {x: Math.ceil(outputSize / 64)},
          programUniforms
        }),
        getShaderSource
      };
    };

export const matMulNBits = (context: ComputeContext, attributes: MatMulNBitsAttributes): void => {
  validateInputs(context.inputs, attributes);
  context.compute(createMatMulNBitsProgramInfo(context.inputs, attributes));
};

export const parseMatMulNBitsAttributes = (attributes: Record<string, unknown>): MatMulNBitsAttributes =>
    createAttributeWithCacheKey(attributes as Omit<MatMulNBitsAttributes, keyof AttributeWithCacheKey>);
