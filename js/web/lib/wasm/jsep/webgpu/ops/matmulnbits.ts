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

export const createBlockwiseMatMulNBitsProgramInfo =
    (inputs: readonly TensorView[], attributes: MatMulNBitsAttributes,
     maxComputeWorkgroupSizes: [number, number, number]): ProgramInfo => {
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
      const outputShape = batchDims.concat([dimAOuter, nBlocksPerCol, dimBOuter]);
      const aComponents = getMaxComponents(attributes.k);
      const bComponents = getMaxComponents(blobSizeInWords);
      const outputSize = ShapeUtil.size(outputShape);
      const workgroupSize = [Math.min(maxComputeWorkgroupSizes[0], outputSize / dimAOuter), 1, 1];
      const programUniforms: ProgramUniform[] = [
        {type: DataType.uint32, data: outputSize / dimAOuter}, {type: DataType.uint32, data: attributes.k},
        {type: DataType.uint32, data: attributes.n}, {type: DataType.uint32, data: attributes.accuracyLevel},
        {type: DataType.uint32, data: attributes.bits}, {type: DataType.uint32, data: attributes.blockSize}
      ];
      const inputShapeTemp = [batchSize, dimAOuter, dimInner / aComponents];
      const bShape = ShapeUtil.convertShape(inputs[1].dims).slice();
      bShape.splice(-1, 1, blobSizeInWords / bComponents);
      programUniforms.push(...createTensorShapeVariables(inputShapeTemp));
      programUniforms.push(...createTensorShapeVariables(bShape));
      programUniforms.push(...createTensorShapeVariables(inputs[2].dims));
      if (inputs.length === 4) {
        programUniforms.push(...createTensorShapeVariables(ShapeUtil.convertShape(inputs[3].dims)));
      }
      const outputShapeTemp = [batchSize, nBlocksPerCol, dimAOuter, dimBOuter];
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
        const output = outputVariable('output', inputs[0].dataType, outputRank);
        const uniforms: UniformsArrayType = [
          {name: 'output_size', type: 'u32'}, {name: 'K', type: 'u32'}, {name: 'N', type: 'u32'},
          {name: 'accuracy_level', type: 'u32'}, {name: 'bits', type: 'u32'}, {name: 'block_size', type: 'u32'}
        ];
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
        const zeroPointsBytesPerCol = Math.floor((nBlocksPerCol + 1) / 2);
        // outputSizePerBatch is the number of elements computed in the output tensor per batch.
        // outputSizePerBatch is computed in number of colums of output tensor produced.
        const outputSizePerBatch = nBlocksPerCol * dimBOuter;
        return `
        ${dequantizeImpl};
        ${ortUnpack8x4snormImpl};
        ${shaderHelper.registerUniforms(uniforms).declareVariables(...inputVariables, output)}
        ${shaderHelper.mainStart([
          workgroupSize[0], workgroupSize[1], workgroupSize[2]
        ])}
          ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.output_size')}
          var output_values = array<${output.type.value}, ${dimAOuter}>(${
            Array.from({length: dimAOuter}, () => `${output.type.value}(0)`).join(', ')});
          var output_indices: ${output.type.indices};
          var a_indices: ${a.type.indices};
          var batch: u32 = global_idx / ${outputSizePerBatch};
          var col: u32 = (global_idx % ${outputSizePerBatch}) / ${nBlocksPerCol};
          var block = (global_idx % ${outputSizePerBatch}) % ${nBlocksPerCol};
          ${output.indicesSet('output_indices', '0', 'batch')};
          ${a.indicesSet('a_indices', '0', 'batch')};
          // Two zero points are packed into one byte when uniforms.bits is 4.
          ${
            zeroPoints ? `
          var zero_point_byte_count: u32 = col * ${zeroPointsBytesPerCol}  + (block >> 0x1u);
          var zero_point_word_index: u32 = zero_point_byte_count >> 0x2u;
          var zero_point_byte_offset: u32 = zero_point_byte_count & 0x3u;
          var zero_point_nibble_offset: u32 = block & 0x1u;
          var zero_point_bits_offset: u32 = (zero_point_byte_offset << 3) + (zero_point_nibble_offset << 2);
          var zero_point_word: u32 = ${zeroPoints.getByOffset('zero_point_word_index')} >> zero_point_bits_offset;` :
                         ''}
          var b_indices: ${b.type.indices};
          ${b.indicesSet('b_indices', '0', 'col')};
          var block_offset: u32 = block * ${attributes.blockSize / aComponents};
          // The scale and zero points are computed per block.
          var scales_index = col * ${nBlocksPerCol} + block;
          let scale = ${scales.getByOffset('scales_index')};
          // The default zero point is 8 for unsigned 4-bit quantization.
          let zero_point = ${dataType}(${zeroPoints ? '(zero_point_word) & 0xFu' : 8.0});
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
                ${a.indicesSet('a_indices', inputRank - 1, 'offset')};
                for (var k: u32 = 0; k < ${dimAOuter}u; k++) {
                  ${a.indicesSet('a_indices', inputRank - 2, 'k')};
                  let a_data = ${a.getByIndices('a_indices')};
                  output_values[k] += ${
            aComponents === 1 ? 'a_data * b_dequantized_values[j]' : 'dot(a_data, b_dequantized_values[j])'};
                                  }
                offset++;
              }
              word_offset += ${8 / aComponents};
            }
          }
          ${output.indicesSet('output_indices', outputRank - 3, 'block')}
          ${output.indicesSet('output_indices', outputRank - 1, 'col')}
          for (var k: u32 = 0u; k < ${dimAOuter}u; k++) {
            ${output.indicesSet('output_indices', outputRank - 2, 'k')};
            ${output.setByIndices('output_indices', 'output_values[k]')}
          }
        }`;
      };
      return {
        name: 'BlockwiseMatMulNBits',
        shaderCache:
            {hint: `${attributes.cacheKey};${inputs.length}`, inputDependencies: Array(inputs.length).fill('rank')},
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
          dispatchGroup: {x: Math.ceil(outputSize / dimAOuter / workgroupSize[0] /* workgroup size */)},
          programUniforms
        }),
        getShaderSource
      };
    };

export const createMatMulNBitsReduceProgramInfo =
    (inputs: readonly TensorView[], attributes: MatMulNBitsAttributes,
     maxComputeWorkgroupSizes: [number, number, number]) => {
      const inputShape = inputs[0].dims;
      const batchDims = inputShape.slice(0, inputShape.length - 3);
      const nBlocksPerCol = inputShape[inputShape.length - 2];
      const dimAOuter = inputShape[inputShape.length - 3];
      const dimBOuter = inputShape[inputShape.length - 1];
      const outputShape = batchDims.concat([dimAOuter, dimBOuter]);
      const outputSize = ShapeUtil.size(outputShape);
      const batchSize = ShapeUtil.size(batchDims);
      const lastDim = inputShape[inputShape.length - 1];
      const components = getMaxComponents(lastDim);
      const inputShapeTmp = [batchSize, nBlocksPerCol, dimAOuter, dimBOuter / components];
      const outputShapeTmp = [batchSize, dimAOuter, dimBOuter / components];
      const workgroupSize = [Math.min(maxComputeWorkgroupSizes[0], outputSize), 1, 1];
      const programUniforms: ProgramUniform[] = [{type: DataType.uint32, data: outputSize}];
      programUniforms.push(...createTensorShapeVariables(inputShapeTmp, outputShapeTmp));
      const getShaderSource = (shaderHelper: ShaderHelper) => {
        const input = inputVariable('input', inputs[0].dataType, inputShapeTmp.length, components);
        const output = outputVariable('output', inputs[0].dataType, outputShapeTmp.length, components);
        return `
          ${shaderHelper.registerUniform('output_size', 'u32').declareVariables(input, output)}
          ${shaderHelper.mainStart([
          workgroupSize[0], workgroupSize[1], workgroupSize[2]
        ])}
            ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.output_size')}
            var output_indices = ${output.offsetToIndices('global_idx')};
            var input_indices = ${input.type.indices}(output_indices[0], 0, output_indices[1], output_indices[2]);
            var output_value: ${output.type.value} = ${output.type.value}(0);
            for (var i: u32 = 0u; i < ${nBlocksPerCol}u; i++) {
              ${input.indicesSet('input_indices', '1', 'i')};
              output_value += ${input.getByIndices('input_indices')};
            }
            ${output.setByIndices('output_indices', 'output_value')}
          }`;
      };
      return {
        name: 'MatMulNBitsReduce',
        shaderCache: {hint: attributes.cacheKey, inputDependencies: ['rank' as const]},
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
          dispatchGroup: {x: Math.ceil(outputSize / components / workgroupSize[0] /* workgroup size */)},
          programUniforms
        }),
        getShaderSource
      };
    };

export const matMulNBits = (context: ComputeContext, attributes: MatMulNBitsAttributes): void => {
  validateInputs(context.inputs, attributes);
  const maxComputeWorkgroupSizes = context.maxComputeWorkgroupSizes();
  const [intermediateResult] = context.compute(
      createBlockwiseMatMulNBitsProgramInfo(context.inputs, attributes, maxComputeWorkgroupSizes), {outputs: [-1]});
  context.compute(
      createMatMulNBitsReduceProgramInfo([intermediateResult], attributes, maxComputeWorkgroupSizes),
      {inputs: [intermediateResult]});
};

export const parseMatMulNBitsAttributes = (attributes: Record<string, unknown>): MatMulNBitsAttributes =>
    createAttributeWithCacheKey(attributes as Omit<MatMulNBitsAttributes, keyof AttributeWithCacheKey>);
