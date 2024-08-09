// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType, getTensorElementSize} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {BroadcastUtil, ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo, ProgramUniform} from '../types';

import {createTensorShapeVariables, getMaxComponents, inputVariable, outputVariable, ShaderHelper, tensorTypeToWsglStorageType, UniformsArrayType} from './common';
import {createMatMulNBitsSharedProgramInfo} from './matmulnbits-shared';
import {createMatMulNBitsSpecialAProgramInfo} from './matmulnbits-special-a';

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
      const dataType = inputs[0].dataType;
      const outputNumber = getMaxComponents(dimAOuter);
      const aComponents = getMaxComponents(attributes.k);
      const bComponents = getMaxComponents(blobSizeInWords);
      const elementSize = getTensorElementSize(dataType)!;
      const workgroupOutputSize = dimAOuter * nBlocksPerCol * elementSize;
      const maxNumberOfComponents = Math.floor(maxComputeWorkgroupStorageSize / workgroupOutputSize);
      const useBlockwiseMatMulNBits = nBlocksPerCol <= maxComputeWorkgroupSizes[0] && maxNumberOfComponents > 0;
      const components = (!useBlockwiseMatMulNBits || maxNumberOfComponents >= 4) ? getMaxComponents(dimBOuter) :
          ((maxNumberOfComponents >= 2) && getMaxComponents(dimBOuter) >= 2)      ? 2 :
                                                                                    1;
      const outputShape = batchDims.concat([dimAOuter, dimBOuter]);
      const outputSize = ShapeUtil.size(outputShape) / components / outputNumber;

      const programUniforms: ProgramUniform[] = useBlockwiseMatMulNBits ?
          [] :
          [{type: DataType.uint32, data: outputSize}, {type: DataType.uint32, data: attributes.blockSize}];
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
        const uniforms: UniformsArrayType = [{name: 'output_size', type: 'u32'}, {name: 'block_size', type: 'u32'}];
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

        const processOneBlock = `
        for (var word: u32 = 0; word < ${blobSizeInWords}; word += ${bComponents}) {
          ${b.indicesSet('b_indices', '2', 'word')};
          let b_data = ${b.getByIndices('b_indices')};
          for (var i: u32 = 0; i < ${bComponents}; i++) {
            let b_value: u32 = ${bComponents === 1 ? 'b_data' : 'b_data[word + i]'};
            let b_mask: u32 = 0x0F0F0F0Fu;
            let b_value_lower: vec4<u32> = unpack4xU8(b_value & b_mask);
            let b_value_upper: vec4<u32> = unpack4xU8((b_value >> 4) & b_mask);
            let b_quantized_values = ${qDqDataType}(${
            Array.from({length: 4}, (_, i) => `${dataType}(b_value_lower[${i}]), ${dataType}(b_value_upper[${i}])`)
                .join(', ')});
            let b_dequantized_values = ${(() => {
          if (aComponents === 1) {
            return `${qDqDataType}(${
                Array.from({length: 8}, (_, i) => `(b_quantized_values[${i}] - zero_point) * scale`).join(', ')});`;
          } else {
            return `(b_quantized_values - ${qDqDataType}(${Array(8).fill('zero_point').join(',')})) * scale;`;
          }
        })()};
            // Number of B elements per 32-bit word is 32/bits = 32/4 = 8
            for (var m: u32 = 0; m < ${useBlockwiseMatMulNBits ? dimAOuter : outputNumber}u; m++) {
              ${a.indicesSet('a_indices', inputRank - 2, useBlockwiseMatMulNBits ? 'm' : `row * ${outputNumber} + m`)};
              ${a.indicesSet('a_indices', inputRank - 1, 'word_offset')};
              var input_offset = ${a.indicesToOffset('a_indices')};
              var a_data: ${qDqDataType};
              for (var j: u32 = 0; j < ${8 / aComponents}; j++) {
                a_data[j] = ${a.getByOffset('input_offset')};
                input_offset++;
              }
              ${useBlockwiseMatMulNBits ? 'workgroup_shared[workgroup_shared_offset + m]' : 'output_values[m]'}${
            components > 1 ? '[c]' : ''} += ${
            Array
                .from(
                    {length: 8 / aComponents},
                    (_, i) => `${
                        aComponents === 1 ? `a_data[${i}] * b_dequantized_values[${i}]` :
                                            `dot(a_data[${i}], b_dequantized_values[${i}])`}`)
                .join(' + ')};
            }
            word_offset += ${8 / aComponents};
          }
        }`;
        const updateZeroPointIndex = zeroPoints ? `
          zero_point_offset += 4;
          if (zero_point_offset == 32) {
            zero_point_offset = 0;
            zero_point_index++;
            zero_point_word = ${zeroPoints.getByOffset('zero_point_index')};
          }` :
                                                  '';

        return useBlockwiseMatMulNBits ? `
        var<workgroup> workgroup_shared: array<${output.type.value}, ${dimAOuter * nBlocksPerCol}>;
        ${shaderHelper.declareVariables(...inputVariables, output)}
        ${shaderHelper.mainStart([
          nBlocksPerCol, 1, 1
        ])}
          var a_indices: ${a.type.indices};
          var block = local_id.x;
          var col = workgroup_id.y;
          var batch = workgroup_id.z;
          ${a.indicesSet('a_indices', '0', 'batch')};
          // Two zero points are packed into one byte when uniforms.bits is 4.
          for (var c: u32 = 0; c < ${components}; c++) {
            let col_times_components_plus_c = col * ${components} + c;
              ${
                                             zeroPoints ? `
            var zero_point_bytes_per_col: u32 = (${nBlocksPerCol} + 1) / 2;
            var zero_point_byte_count: u32 = col_times_components_plus_c * zero_point_bytes_per_col + (block >> 0x1u);
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
            var word_offset: u32 = block * ${attributes.blockSize / aComponents};
            var workgroup_shared_offset: u32 = block * ${dimAOuter};
            ${processOneBlock}
          }
          workgroupBarrier();
          var output_indices: ${output.type.indices};
          var elements_per_thread: u32 = ${Math.ceil(dimAOuter / nBlocksPerCol)};
          ${output.indicesSet('output_indices', '0', 'batch')};
          ${output.indicesSet('output_indices', outputRank - 1, 'col')};
          ${output.indicesSet('output_indices', outputRank - 2, 'local_id.x * elements_per_thread')};
          var output_offset = ${output.indicesToOffset('output_indices')};
          for (var m: u32 = 0u; m < elements_per_thread; m++) {
            var row = m + local_id.x * elements_per_thread;
            if (row < ${dimAOuter}) {
              var output_value: ${output.type.value} = ${output.type.value}(0);
              var workgroup_shared_offset: u32 = row;
              for (var b: u32 = 0u; b < ${nBlocksPerCol}u; b++) {
                output_value += workgroup_shared[workgroup_shared_offset];
                workgroup_shared_offset += ${dimAOuter};
              }
              ${output.setByOffset('output_offset', 'output_value')};
              output_offset += ${dimBOuter / components};
            }
          }
        }` :
                                         `
        ${shaderHelper.registerUniforms(uniforms).declareVariables(...inputVariables, output)}
        ${shaderHelper.mainStart()}
          ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.output_size')}
          var output_values: array<${output.type.value}, ${outputNumber}>;
          var output_indices = ${output.offsetToIndices('global_idx')};
          var col = ${output.indicesGet('output_indices', outputRank - 1)};
          var row = ${output.indicesGet('output_indices', outputRank - 2)};
          var a_indices: ${a.type.indices} = output_indices;
          // Two zero points are packed into one byte because uniforms.bits <= 4.
          // zero_point_offset is either 0 or 4. It is bit offset within one byte.
          // TODO support zero_point_offset for bits > 4
          ${
                                             zeroPoints ? `
          var zero_point_abs_offset = col * ${components} * ((${nBlocksPerCol} + 1) / 2);
          var zero_point_index: u32 = zero_point_abs_offset / 4;
          var zero_point_word: u32 = ${zeroPoints.getByOffset('zero_point_index')};
          var zero_point_offset: u32 = (zero_point_abs_offset % 4) * 8;` :
                                                          ''}
          var scale_index = col * ${nBlocksPerCol * components};
          var b_indices: ${b.type.indices};
          for (var c: u32 = 0; c < ${components}; c++) {
            ${b.indicesSet('b_indices', '0', `col * ${components} + c`)};
            var block_offset: u32 = 0;
            for (var block: u32 = 0; block < ${nBlocksPerCol}; block++) {
              // The scale and zero points are computed per block.
              let scale = ${scales.getByOffset('scale_index')};
              // The default zero point is 8 for unsigned 4-bit quantization.
              let zero_point = ${dataType}(${zeroPoints ? 'extractBits(zero_point_word, zero_point_offset, 4)' : 8.0});
              ${b.indicesSet('b_indices', '1', 'block')};
              var word_offset: u32 = block_offset;
              ${processOneBlock}
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
              ${output.indicesSet('output_indices', outputRank - 2, `${outputNumber} * row + k`)};
              ${output.setByIndices('output_indices', 'output_values[k]')}
            }
        }`;
      };
      return {
        name: useBlockwiseMatMulNBits ? 'BlockwiseMatMulNBits' : 'MatMulNBits',
        shaderCache: {
          hint: `${attributes.cacheKey};${dimAOuter};${dataType};${inputs.length}`,
          inputDependencies: Array(inputs.length).fill('rank')
        },
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType}],
          name: useBlockwiseMatMulNBits ? 'BlockwiseMatMulNBits' : 'MatMulNBits',
          dispatchGroup: useBlockwiseMatMulNBits ? {x: 1, y: Math.ceil(dimBOuter / components), z: batchSize} :
                                                   {x: Math.ceil(outputSize / 64 /* workgroup size */)},
          programUniforms
        }),
        getShaderSource
      };
    };

export const matMulNBits = (context: ComputeContext, attributes: MatMulNBitsAttributes): void => {
  validateInputs(context.inputs, attributes);

  const N = context.inputs[1].dims[0];
  const K = context.inputs[0].dims[context.inputs[0].dims.length - 1];
  if (context.inputs.length === 3 && attributes.bits === 4 && N % 4 == 0 && N >= 32 && K % 32 === 0) {
    const outputShape = BroadcastUtil.calcShape(context.inputs[0].dims, [attributes.k, attributes.n], true);
    if (!outputShape) {
      throw new Error('Can\'t use matmul on the given tensors');
    }
    if (context.inputs[0].dims.length === 3 && context.inputs[0].dims[0] === 1 && context.inputs[0].dims[1] === 1 &&
        context.inputs[0].dims[2] >= 1024) {
      context.compute(createMatMulNBitsSpecialAProgramInfo(context.inputs, attributes, outputShape));
    } else {
      context.compute(createMatMulNBitsSharedProgramInfo(context.inputs, attributes, outputShape));
    }
  } else {
    const maxComputeWorkgroupSizes: [number, number, number] = context.getMaxComputeWorkgroupSizes();
    const maxComputeWorkgroupStorageSize = context.getMaxComputeWorkgroupStoragesize();
    context.compute(createMatMulNBitsProgramInfo(
        context.inputs, attributes, maxComputeWorkgroupSizes, maxComputeWorkgroupStorageSize));
  }
};

export const parseMatMulNBitsAttributes = (attributes: Record<string, unknown>): MatMulNBitsAttributes =>
    createAttributeWithCacheKey(attributes as Omit<MatMulNBitsAttributes, keyof AttributeWithCacheKey>);
