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
      const inputShape = inputs[0].dims;
      const aRank = inputShape.length;
      const outputShape = inputShape.slice(0, aRank - 1).concat(attributes.n);
      const m = inputShape[aRank - 2];
      const blobSize = attributes.blockSize / 8 * attributes.bits;
      const blobSizeInWords = blobSize / 4;
      const outputNumber = getMaxComponents(m);
      const components = getMaxComponents(attributes.n);
      const aComponents = getMaxComponents(attributes.k);
      const bComponents = getMaxComponents(blobSizeInWords);
      const outputSize = ShapeUtil.size(outputShape) / components / outputNumber;
      const programUniforms: ProgramUniform[] = [
        {type: DataType.uint32, data: outputSize}, {type: DataType.uint32, data: attributes.k},
        {type: DataType.uint32, data: attributes.n}, {type: DataType.uint32, data: attributes.accuracyLevel},
        {type: DataType.uint32, data: attributes.bits}, {type: DataType.uint32, data: attributes.blockSize}
      ];
      const aShape = inputShape.slice();
      aShape.splice(-1, 1, attributes.k / aComponents);
      const bShape = ShapeUtil.convertShape(inputs[1].dims).slice();
      bShape.splice(-1, 1, blobSizeInWords / bComponents);
      programUniforms.push(...createTensorShapeVariables(aShape));
      programUniforms.push(...createTensorShapeVariables(bShape));
      programUniforms.push(...createTensorShapeVariables(inputs[2].dims));
      if (inputs.length === 4) {
        programUniforms.push(...createTensorShapeVariables(ShapeUtil.convertShape(inputs[3].dims)));
      }
      const oShape = outputShape.slice();
      oShape.splice(-1, 1, attributes.n / components);
      programUniforms.push(...createTensorShapeVariables(oShape));
      const getShaderSource = (shaderHelper: ShaderHelper) => {
        const a = inputVariable('a', inputs[0].dataType, aShape.length, aComponents);
        const b = inputVariable('b', DataType.uint32, bShape.length, bComponents);
        const scales = inputVariable('scales', inputs[2].dataType, inputs[2].dims.length);
        const inputVariables = [a, b, scales];
        const zeroPoints =
            inputs.length === 4 ? inputVariable('zero_points', DataType.uint32, inputs[3].dims.length) : undefined;
        if (zeroPoints) {
          inputVariables.push(zeroPoints);
        }
        const output = outputVariable('output', inputs[0].dataType, outputShape.length, components);
        const uniforms: UniformsArrayType = [
          {name: 'output_size', type: 'u32'}, {name: 'K', type: 'u32'}, {name: 'N', type: 'u32'},
          {name: 'accuracy_level', type: 'u32'}, {name: 'bits', type: 'u32'}, {name: 'block_size', type: 'u32'}
        ];
        const nBlocksPerCol = Math.floor((attributes.k + attributes.blockSize - 1) / attributes.blockSize);
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

        const dequantizeImpl = `
        fn dequantize(quantized: ${qDqDataType}, zero_point: ${dataType}, scale: ${dataType}) -> ${qDqDataType} {
          ${(() => {
          if (aComponents === 1) {
            return `var dequantized = ${qDqDataType}(${
                Array.from({length: 8}, (_, i) => `(quantized[${i}] - zero_point) * scale`).join(', ')});
              return dequantized;`;
          } else {
            return `var zero_points: ${qDqDataType} = ${qDqDataType}(${Array(8).fill('zero_point').join(',')});
              return (quantized - zero_points) * scale;`;
          }
        })()}
        }`;
        const ortUnpack8x4snormImpl = `
        fn ortUnpack8x4snorm(value: u32) -> ${qDqDataType} {
          return ${qDqDataType}(${
            Array.from({length: 8}, (_, i) => `${dataType}((value >> ${(i * 4).toString()}) & 0xFu)`).join(', ')});
        }`;

        const updateZeroPointIndex = zeroPoints ? `
          zero_point_offset += 4;
          if (zero_point_offset == 32) {
            zero_point_offset = 0;
            zero_point_index++;
            zero_point_word = ${zeroPoints.getByOffset('zero_point_index')};
          }` :
                                                  '';

        return `
        ${dequantizeImpl};
        ${ortUnpack8x4snormImpl};
        ${shaderHelper.registerUniforms(uniforms).declareVariables(...inputVariables, output)}
        ${shaderHelper.mainStart()}
          ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.output_size')}
          var output_values: array<${output.type.value}, ${outputNumber}>;
          var output_indices = ${output.offsetToIndices('global_idx')};
          var n = ${output.indicesGet('output_indices', aRank - 1)};
          var m = ${output.indicesGet('output_indices', aRank - 2)};
          var row = ${outputNumber} * m;
          var col = ${components} * n;
          var a_indices: ${a.type.indices} = output_indices;
          // Two zero points are packed into one byte when uniforms.bits is 4.
          ${
            zeroPoints ? `
          var zero_point_index: u32 = col * ((${nBlocksPerCol} + 1) / 2) / 4;
          var zero_point_word: u32 = ${zeroPoints.getByOffset('zero_point_index')};
          var zero_point_byte_offset: u32 = (col * ((${nBlocksPerCol} + 1) / 2)) % 4;
          var zero_point_offset: u32 = 8 * zero_point_byte_offset;` :
                         ''}
          var scale_index = col * ${nBlocksPerCol};
          var b_indices: ${b.type.indices};
          for (var c: u32 = 0; c < ${components}; c++) {
            ${b.indicesSet('b_indices', '0', 'col + c')};
            var block_offset: u32 = 0;
            for (var block: u32 = 0; block < ${nBlocksPerCol}; block++) {
              // The scale and zero points are computed per block.
              let scale = ${scales.getByOffset('scale_index')};
              // The default zero point is 8 for unsigned 4-bit quantization.
              let zero_point = ${dataType}(${zeroPoints ? 'extractBits(zero_point_word, zero_point_offset, 4)' : 8.0});
              ${b.indicesSet('b_indices', '1', 'block')};
              var word_offset: u32 = block_offset;
              for (var word: u32 = 0; word < ${blobSizeInWords}; word += ${bComponents}) {
                ${b.indicesSet('b_indices', '2', 'word')};
                let b_data = ${b.getByIndices('b_indices')};
                for (var i: u32 = 0; i < ${bComponents}; i++) {
                  let b_value = ${bComponents === 1 ? 'b_data' : 'b_data[word + i]'};
                  let b_quantized_values: ${qDqDataType} = ortUnpack8x4snorm(b_value);
                  let b_dequantized_values = dequantize(b_quantized_values, zero_point, scale);
                  // Number of B elements per 32-bit word is 32/bits = 32/4 = 8
                  var offset: u32 = word_offset;
                  for (var j: u32 = 0; j < 8; j += ${aComponents}) {
                    ${a.indicesSet('a_indices', aRank - 1, 'offset')};
                    for (var k: u32 = 0; k < ${outputNumber}u; k++) {
                      ${a.indicesSet('a_indices', aRank - 2, 'row + k')};
                      let a_data = ${a.getByIndices('a_indices')};
                      output_values[k]${components > 1 ? '[c]' : ''} += ${
            aComponents === 1 ? 'a_data * b_dequantized_values[j]' : 'dot(a_data, b_dequantized_values[j])'};
                    }
                    offset++;
                  }
                  word_offset += ${8 / aComponents};
                }
              }
              scale_index++;
              ${updateZeroPointIndex}
              block_offset += uniforms.block_size / ${aComponents};
            }
            // Drop the trailing 4 bits if the zero_poit_offset is not a byte boundary to align with the next byte.
            ${
            zeroPoints ? `if (zero_point_offset % 8 > 0) {
                ${updateZeroPointIndex}
              }` :
                         ''}
          }
          for (var k: u32 = 0u; k < ${outputNumber}u; k++) {
            ${output.indicesSet('output_indices', aRank - 2, `${outputNumber + ' * m + k'}`)};
            ${output.setByIndices('output_indices', 'output_values[k]')}
          }
        }`;
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
