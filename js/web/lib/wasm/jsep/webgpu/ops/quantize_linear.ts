// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo, ProgramUniform} from '../types';

import {createTensorShapeVariables, inputVariable, outputVariable, ShaderHelper, tensorTypeToWsglStorageType, UniformsArrayType} from './common';

export interface DequantizeLinerAttributes extends AttributeWithCacheKey {
  axis: number;
  blockSize: number;
}

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (inputs.length < 2 || inputs.length > 3) {
    throw new Error('DequantizeLinear requires 2 or 3 inputs.');
  }
  if (inputs.length === 3 && inputs[1].dims === inputs[2].dims) {
    throw new Error('x-scale and x-zero-point must have the same shape');
  }
  if (inputs.length === 3 && inputs[0].dataType !== inputs[2].dataType) {
    throw new Error('x and x-zero-point must have the same data type');
  }
  if (inputs[0].dataType === DataType.int32 && inputs.length > 2) {
    throw new Error('In the case of dequantizing int32 there is no zero point.');
  }
  if (inputs[1].dims.length !== 0 && inputs[1].dims.length !== 1 && inputs[1].dims.length !== inputs[0].dims.length) {
    throw new Error('scale input must be a scalar, a 1D tensor, or have the same rank as the input tensor');
  }
};

const createDequantizeLinearProgramInfo =
    (inputs: readonly TensorView[], attributes: DequantizeLinerAttributes): ProgramInfo => {
      const axis = ShapeUtil.normalizeAxis(attributes.axis, inputs[0].dims.length);
      const inputType = inputs[0].dataType;
      const outputShape = inputs[0].dims;   // output shape is same as the input shape
      const dataType = inputs[1].dataType;  // output type is same as the the scale input type
      const outputSize = ShapeUtil.size(outputShape);
      const uniforms: UniformsArrayType = [{name: 'output_size', type: 'u32'}, {name: 'axis', type: 'u32'}];
      const isPacked = inputType === DataType.int8 || inputType === DataType.uint8;
      const inputShape = isPacked ? ShapeUtil.convertShape(inputs[0].dims).slice() : inputs[0].dims;
      const input = inputVariable('input', DataType.uint32, inputShape.length);
      const scale = inputVariable('scale', inputs[1].dataType, inputs[1].dims.length);
      const zeroPoint =
          inputs.length > 2 ? inputVariable('zero_point', DataType.uint32, inputs[2].dims.length) : undefined;
      const output = outputVariable('output', dataType, outputShape.length);
      const inputVariables = [input, scale];
      if (zeroPoint) {
        inputVariables.push(zeroPoint);
      }
      const programUniforms: ProgramUniform[] = [
        {type: DataType.uint32, data: outputSize}, {type: DataType.uint32, data: axis},
        ...createTensorShapeVariables(...inputs.map((t) => t.dims), outputShape)
      ];

      const getShaderSource = (shaderHelper: ShaderHelper) => `
      ${shaderHelper.registerUniforms(uniforms).declareVariables(...inputVariables, output)}
      ${shaderHelper.mainStart()}
          ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.output_size')}
          // Set input x
          ${(() => {
        if (isPacked) {
          const isSigned = inputType === DataType.uint8;
          return `
              let input = ${input.getByOffset('global_idx/4')};
              let x_vec: vec4<${isSigned ? 'i32' : 'u32'}> = ${isSigned ? 'unpack4xI8(input)' : 'unpack4xU8(input)'};
              let x = x_vec[global_idx % 4];`;
        } else {
          return `let x = ${input.getByOffset('global_idx')};`;
        }
      })()};

          // Set scale input
          ${(() => {
        const shape = inputs[1].dims;
        if (shape.length === 0 || (shape.length === 1 && shape[0] === 1)) {
          // scale input is a scalar
          return `
              let scale = ${scale.getByOffset('0')}`;
        } else if (shape.length === 1) {
          // scale input is a 1D tensor
          return `
                  let input_indices = ${input.offsetToIndices('global_idx')};
                  let input_index = ${input.indicesGet('input_indices', 'uniforms.axis')};
                  let scale_indices: ${scale.type.indices};
                  ${scale.indicesSet('scale_indices', 'input_index', '0')};
                  scale = ${scale.getByOffset('scale_index')};`;
        } else {
          return `
              scale = ${scale.getByOffset('global_idx')};`;
        }
      })()};

          // Set zero-point input
          ${(() => {
        if (zeroPoint) {
          const isSigned = inputType === DataType.int8;
          const shape = inputs[2].dims;
          if (shape.length === 0 || (shape.length === 1 && shape[0] === 1)) {
            // zero-point input is a scalar
            if (isPacked) {
              return `
              let zero_point_input = ${zeroPoint.getByOffset('0')};
              let zero_point_vec: vec4<${isSigned ? 'i32' : 'u32'}> =  ${
                  isSigned ? 'unpack4xI8(zero_point_input)' : 'unpack4xU8(zero_point_input)'};
              let zero_point = zero_point_vec[0]`;
            } else {
              return `let zero_point = ${zeroPoint.getByOffset('0')}`;
            }
            // return `let zero_point = ${isSigned ? 'i32' : 'u32'}(${zeroPoint.getByOffset('0')});`;
          } else if (shape.length === 1) {
            // zero-point input is a 1D tensor
            if (isPacked) {
              return `
              let input_indices = ${input.offsetToIndices('global_idx')};
              let input_index = ${input.indicesGet('input_indices', 'uniforms.axis')};
              let zero_point_indices: ${zeroPoint.type.indices};
              ${zeroPoint.indicesSet('zero_point_indices', 'input_index', '0')};
              let zero_point_input = ${zeroPoint.getByOffset('zero_point_index')};
              let zero_point_vec: vec4<${isSigned ? 'i32' : 'u32'}> =  ${
                  isSigned ? 'unpack4xI8(zero_point_input)' : 'unpack4xU8(zero_point_input)'};
              let zero_point = zero_point_vec[global_idx % 4]`;
            } else {
              return `
              let input_indices = ${input.offsetToIndices('global_idx')};
              let input_index = ${input.indicesGet('input_indices', 'uniforms.axis')};
              let zero_point_indices: ${zeroPoint.type.indices};
              ${zeroPoint.indicesSet('zero_point_indices', 'input_index', '0')};
              let zero_point = ${zeroPoint.getByOffset('zero_point_index')};`;
            }
          } else {
            // blocked quantization
            if (isPacked) {
              return `
              let zero_point_input = ${input.getByOffset('global_idx/4')};
              let zero_point_vec: vec4<${isSigned ? 'i32' : 'u32'} = ${
                  isSigned ? 'unpack4xI8(zero_point_input)' : 'unpack4xU8(zero_point_input)'};
              let zero_point = zero_point_vec[global_idx % 4];`;
            } else {
              return `let zero_point = ${isSigned ? 'i32' : 'u32'}(${zeroPoint.getByOffset('0')});`;
            }
          }
        } else {
          return 'let zero_point = 0;';
        }
      })()};
      // Compute and write output
      ${output.setByOffset('global_idx', `${tensorTypeToWsglStorageType(dataType)}(x - zero_point) * scale`)};
      }`;
      return {
        name: 'DequantizeLinear',
        shaderCache:
            {hint: attributes.cacheKey, inputDependencies: zeroPoint ? ['rank', 'rank', 'rank'] : ['rank', 'rank']},
        getShaderSource,
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType}],
          dispatchGroup: {x: Math.ceil(outputSize / 64), y: 1, z: 1},
          programUniforms
        })
      };
    };

export const dequantizeLinear = (context: ComputeContext, attributes: DequantizeLinerAttributes): void => {
  validateInputs(context.inputs);
  context.compute(createDequantizeLinearProgramInfo(context.inputs, attributes));
};

export const parseDequantizeLinearAttributes = (attributes: Record<string, unknown>): DequantizeLinerAttributes =>
    createAttributeWithCacheKey({axis: attributes.axis as number, blockSize: attributes.blockSize as number});
