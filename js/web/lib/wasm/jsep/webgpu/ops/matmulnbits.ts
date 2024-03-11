// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo, ProgramUniform} from '../types';

import {createTensorShapeVariables, getMaxComponents, inputVariable, outputVariable, ShaderHelper, tensorTypeToWsglStorageType, UniformsArrayType} from './common';

//  TODO support quantization bits not equal to 4
export interface MatMulNBitsAttributes extends AttributeWithCacheKey {
  K: number;
  N: number;
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
  if (a.dims[aRank - 1] !== attributes.K) {
    throw new Error('The last dim of input shape does not match the k value');
  }
  const nBlocksPerCol = Math.floor((attributes.K + attributes.blockSize - 1) / attributes.blockSize);
  const blobSize = attributes.blockSize / 8 * attributes.bits;
  const b = inputs[1];
  if (!ShapeUtil.areEqual(b.dims, [attributes.N, nBlocksPerCol, blobSize])) {
    throw new Error('The second inputs must be 3D tensor with shape N X nBlocksPerCol X blobSize');
  }
  const scales = inputs[2];
  const scalesShape = scales.dims;
  if (ShapeUtil.size(scalesShape) !== attributes.N * nBlocksPerCol) {
    throw new Error('scales input size error.');
  }
  if (inputs.length === 4) {
    const zeroPoints = inputs[3];
    const zeroPointsShape = zeroPoints.dims;
    const expectedZeroPointsSize =
        attributes.bits > 4 ? (attributes.N * nBlocksPerCol) : attributes.N * Math.floor((nBlocksPerCol + 1) / 2);
    if (ShapeUtil.size(zeroPointsShape) !== expectedZeroPointsSize) {
      throw new Error('zeroPoints input size error.');
    }
  }
};

export const createMatMulNBitsProgramInfo =
    (inputs: readonly TensorView[], attributes: MatMulNBitsAttributes): ProgramInfo => {
      const inputShape = inputs[0].dims;
      const aRank = inputShape.length;
      const outputShape = inputShape.slice(0, aRank - 1).concat(attributes.N);
      const M = inputShape[aRank - 2];
      const blobSize = attributes.blockSize / 8 * attributes.bits;
      const blobSizeInWords = blobSize / 4;
      const outputNumber = getMaxComponents(M);
      const components = 1;  // getMaxComponents(attributes.n);
      const aComponents = getMaxComponents(attributes.K);
      const bComponents = getMaxComponents(blobSizeInWords);
      const zComponents = 1;  // getMaxComponents(attributes.N / 8);
      const outputSize = ShapeUtil.size(outputShape) / components / outputNumber;
      const programUniforms: ProgramUniform[] = [
        {type: DataType.uint32, data: outputSize}, {type: DataType.uint32, data: attributes.K},
        {type: DataType.uint32, data: attributes.N}, {type: DataType.uint32, data: attributes.accuracyLevel},
        {type: DataType.uint32, data: attributes.bits}, {type: DataType.uint32, data: attributes.blockSize}
      ];
      const getShaderSource = (shaderHelper: ShaderHelper) => {
        const aShape = inputs[0].dims.slice();
        aShape.splice(-1, 1, attributes.k / aComponents);
        const a = inputVariable('a', inputs[0].dataType, aShape, aComponents);
        const bShape = ShapeUtil.convertShape(inputs[1].dims).slice();
        bShape.splice(-1, 1, blobSizeInWords / bComponents);
        const b = inputVariable('b', DataType.uint32, bShape, bComponents);
        const scales = inputVariable('scales', inputs[2].dataType, inputs[2].dims);
        const inputVariables = [a, b, scales];
        const zeroPoints = inputs.length === 4 ?
            inputVariable('zero_points', DataType.uint32, inputs[3].dims, zComponents) :
            undefined;
        if (zeroPoints) {
          inputVariables.push(zeroPoints);
        }
        const output = outputVariable('output', inputs[0].dataType, outputShape, components);
        const uniforms: UniformsArrayType = [
          {name: 'output_size', type: 'u32'}, {name: 'K', type: 'u32'}, {name: 'N', type: 'u32'},
          {name: 'accuracy_level', type: 'u32'}, {name: 'bits', type: 'u32'}, {name: 'block_size', type: 'u32'}
        ];
        const nBlocksPerCol = Math.floor((attributes.K + attributes.blockSize - 1) / attributes.blockSize);
        const dataType = tensorTypeToWsglStorageType(inputs[0].dataType);
        const dequantizeArrayReturnType = (() => {
          switch (aComponents) {
            case 1:
              return `array<${dataType}, 8>`;
            case 2:
              return `array<vec2<${dataType}>, 4>`;
            case 4:
              return `array<vec4<${dataType}>, 2>`;
            default:
              throw new Error(`${aComponents}-component is not supported.`);
          }
        })();
        const dequantizeArrayImpl =
            (() => `fn dequantize_array(quantized_data: array<${dataType}, 8>, zero_point: ${dataType}, scale: ${
                 dataType}) -> ${dequantizeArrayReturnType} {
          var result: ${dequantizeArrayReturnType};
          ${(() => {
               switch (aComponents) {
                 case 1:
                   return `
              for (var i: u32 = 0; i < 8; i++) {
                result[i] = dequantize(quantized_data[i], zero_point, scale);
              }`;
                 case 2:
                   return `
              for (var i: u32 = 0; i < 4; i++) {
                let dequantized0 = dequantize(quantized_data[i*2], zero_point, scale);
                let dequantized1 = dequantize(quantized_data[i*2+1], zero_point, scale);
                result[i] = vec2<${dataType}>(dequantized0, dequantized1);
              }`;
                 case 4:
                   return `
              for (var i: u32 = 0; i < 2; i++) {
                let dequantized0 = dequantize(quantized_data[i*4], zero_point, scale);
                let dequantized1 = dequantize(quantized_data[i*4+1], zero_point, scale);
                let dequantized2 = dequantize(quantized_data[i*4+2], zero_point, scale);
                let dequantized3 = dequantize(quantized_data[i*4+3], zero_point, scale);
                result[i] = vec4<${dataType}>(dequantized0, dequantized1, dequantized2, dequantized3);
              }`;
                 default:
                   throw new Error(`${aComponents}-component is not supported.`);
               }
             })()}
          return result;
        }`)();

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

        fn dequantize(value: ${dataType}, zero_point: ${dataType}, scale: ${dataType}) -> ${dataType} {
          return (value - zero_point) * scale;
        }

        ${dequantizeArrayImpl};

        ${shaderHelper.registerUniforms(uniforms).declareVariables(...inputVariables, output)}
        ${shaderHelper.mainStart()}
          ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.output_size')}
          var output_values: array<${output.type.value}, ${outputNumber}>;
          var output_indices = ${output.offsetToIndices('global_idx')};
          var n = ${output.indicesGet('output_indices', aRank - 1)};
          var m = ${output.indicesGet('output_indices', aRank - 2)};
          var a_indices: ${a.type.indices} = output_indices;
          // Two zero points are packed into one byte because uniforms.bits <= 4.
          // zero_point_offset is either 0 or 4. It is bit offset within one byte.
          // TODO support zero_point_offset for bits > 4
          ${
            zeroPoints ? `
            var zero_point_index: u32 = n * ((${nBlocksPerCol} + 1) / 2) / 4;
            var zero_point_word: u32 = ${zeroPoints.getByOffset('zero_point_index')};
            var zero_point_offset: u32 = 0;` :
                         ''}
          var scale_index = n * ${nBlocksPerCol};
          var b_indices: ${b.type.indices};
          ${b.indicesSet('b_indices', '0', 'n')};
          var block_offset: u32 = 0;
          for (var block: u32 = 0; block < ${nBlocksPerCol}; block++) {
            // The scale and zero points are computed per block.
            let scale = ${scales.getByOffset('scale_index')};
            // The default zero point is 8 for unsigned 4-bit quantization.
            let zero_point: ${dataType} = ${
            zeroPoints ? `${dataType}(extractBits(zero_point_word, zero_point_offset, 4))` : 8.0};
            ${b.indicesSet('b_indices', '1', 'block')};
            var word_offset: u32 = block_offset;
            for (var word: u32 = 0; word < ${blobSizeInWords}; word += ${bComponents}) {
              ${b.indicesSet('b_indices', '2', 'word')};
              let b_data = ${b.getByIndices('b_indices')};
              for (var i: u32 = 0; i < ${bComponents}; i++) {
                let b_value = ${bComponents === 1 ? 'b_data' : 'b_data[word + i]'};
                let b_quantized_values: array<${dataType}, 8> = ortUnpack8x4snorm(b_value);
                let b_dequantized_values = dequantize_array(b_quantized_values, zero_point, scale);
                // Number of B elements per 32-bit word is 32/bits = 32/4 = 8
                var offset: u32 = word_offset;
                for (var j: u32 = 0; j < 8/${aComponents}; j++) {
                  ${a.indicesSet('a_indices', aRank - 1, `offset/${aComponents}`)};
                  for (var k: u32 = 0; k < ${outputNumber}; k++) {
                    ${a.indicesSet('a_indices', aRank - 2, `m * ${outputNumber} + k`)};
                    let a_data = ${a.getByIndices('a_indices')};
                    output_values[k] += ${
            aComponents === 1 ? 'a_data * b_dequantized_values[j]' : `dot(a_data, b_dequantized_values[j])`};
                  }
                  offset += ${aComponents};
                }
                word_offset += 8;
              }
            }
            scale_index++;
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
          for (var k: u32 = 0u; k < ${outputNumber}u; k++) {
            ${output.indicesSet('output_indices', aRank - 2, `${outputNumber + ' * m + k'}`)};
            ${output.setByIndices('output_indices', 'output_values[k]')}
          }
        }
        `;
      };
      return {
        name: 'MatMulNBits',
        shaderCache:
            {hint: `${attributes.cacheKey};${inputs.length}`, inputDependencies: Array(inputs.length).fill('rank')},
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
          dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)},
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
