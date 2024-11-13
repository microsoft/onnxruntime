// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { DataType } from '../../../wasm-common';
import { TensorView } from '../../tensor-view';
import { ShapeUtil } from '../../util';
import { AttributeWithCacheKey, createAttributeWithCacheKey } from '../attribute-with-cache-key';
import { ComputeContext, ProgramInfo, ProgramUniform } from '../types';

import {
  atomicOutputVariable,
  createTensorShapeVariables,
  inputVariable,
  outputVariable,
  ShaderHelper,
} from './common';

export interface ScatterNDAttributes extends AttributeWithCacheKey {
  reduction: string;
}

type ReductionType = 'i32' | 'u32' | 'f32';

const atomicReductionSnippet = (reduction: string, ptr: string, v: string, type: ReductionType) => {
  if (reduction !== 'none' && type !== 'i32' && type !== 'u32' && type !== 'f32') {
    throw new Error(`Input ${type} is not supported with reduction ${reduction}.`);
  }

  const floatStart = `{
                var oldValue = 0;
                loop {
                  let newValueF32 =`;
  const floatEnd = `;
                  let newValue = bitcast<i32>(newValueF32);
                  let res = atomicCompareExchangeWeak(&${ptr}, oldValue, newValue);
                  if res.exchanged {
                    break;
                  }
                  oldValue = res.old_value;
                }
              }`;

  switch (reduction) {
    case 'none':
      return `${ptr}=${v};`;
    case 'add':
      if (type === 'i32' || type === 'u32') {
        return `atomicAdd(&${ptr}, bitcast<${type}>(${v}));`;
      } else {
        // atomicAdd only supports uint/int type. For float, we use
        // atomicCompareExchangeWeak to simulate.
        return `
              ${floatStart}bitcast<${type}>(oldValue) + (${v})${floatEnd}`;
      }
    case 'max':
      if (type === 'i32' || type === 'u32') {
        return `atomicMax(&${ptr}, bitcast<${type}>(${v}));`;
      } else {
        // atomicMax only supports uint/int type. For float, we use
        // atomicCompareExchangeWeak to simulate.
        return `
                ${floatStart}max(bitcast<f32>(oldValue), (${v}))${floatEnd}`;
      }
    case 'min':
      if (type === 'i32' || type === 'u32') {
        return `atomicMin(&${ptr}, bitcast<${type}>(${v}));`;
      } else {
        // atomicMin only supports uint/int type. For float, we use
        // atomicCompareExchangeWeak to simulate.
        return `${floatStart}min(bitcast<${type}>(oldValue), (${v}))${floatEnd}`;
      }
    case 'mul':
      // atomicMul is not supported, we use atomicCompareExchangeWeak to simulate.
      return `${floatStart}(bitcast<${type}>(oldValue) * (${v}))${floatEnd}`;

    default:
      throw new Error(`Reduction ${reduction} is not supported.`);
  }
};

const createScatterNDProgramInfo = (inputs: readonly TensorView[], attributes: ScatterNDAttributes): ProgramInfo => {
  const inputShape = inputs[0].dims;
  const indicesShape = inputs[1].dims;
  const outputShape = inputShape;
  // TODO: support bool with components 4.
  const components = 1;
  const outputSize = Math.ceil(ShapeUtil.size(indicesShape) / components);
  const lastIndexDimension = indicesShape[indicesShape.length - 1];
  const numUpdatesElements = ShapeUtil.sizeFromDimension(inputShape, lastIndexDimension);

  const programUniforms: ProgramUniform[] = [
    { type: DataType.uint32, data: outputSize },
    { type: DataType.uint32, data: lastIndexDimension },
    { type: DataType.uint32, data: numUpdatesElements },
    ...createTensorShapeVariables(inputs[1].dims, inputs[2].dims, outputShape),
  ];

  const getShaderSource = (shaderHelper: ShaderHelper) => {
    const indices = inputVariable('indices', inputs[1].dataType, inputs[1].dims.length);
    const updates = inputVariable('updates', inputs[2].dataType, inputs[2].dims.length, components);
    const output =
      attributes.reduction !== 'none' && attributes.reduction !== ''
        ? atomicOutputVariable('output', inputs[0].dataType, outputShape.length)
        : outputVariable('output', inputs[0].dataType, outputShape.length, components);

    return `
      ${shaderHelper
        .registerUniform('output_size', 'u32')
        .registerUniform('last_index_dimension', 'u32')
        .registerUniform('num_updates_elements', 'u32')
        .declareVariables(indices, updates, output)}
      ${shaderHelper.mainStart()}
        ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.output_size')}
  var data_offset = 0u;
  let indices_start = uniforms.last_index_dimension * global_idx;
  let indices_end = indices_start + uniforms.last_index_dimension;
  for (var i = indices_start; i < indices_end; i++) {
    var index = i32(indices[i].x);
    ${
      inputs[0].dims.length === 1
        ? `
    let element_count_dim = uniforms.output_strides;
    let dim_value = uniforms.output_shape;`
        : `
    let element_count_dim = uniforms.output_strides[i - indices_start];
    let dim_value = uniforms.output_shape[i - indices_start + uniforms.last_index_dimension];`
    }
    if (index >= 0) {
      if (index >= i32(dim_value)) {
        index = i32(dim_value - 1);
      }
    } else {
      if (index < -i32(dim_value)) {
        index = 0;
      } else {
        index += i32(dim_value);
      }
    }
    data_offset += u32((u32(index) * element_count_dim));
  }

  for (var i = 0u; i < uniforms.num_updates_elements; i++) {
    let value = updates[uniforms.num_updates_elements * global_idx + i];
    ${atomicReductionSnippet(
      attributes.reduction,
      'output[data_offset + i]',
      'value',
      output.type.value as ReductionType,
    )}
  }

      }`;
  };
  return {
    name: 'ScatterND',
    shaderCache: {
      hint: `${attributes.cacheKey}_${attributes.reduction}`,
      inputDependencies: ['rank', 'rank'],
    },
    getRunData: () => ({
      outputs: [{ dims: outputShape, dataType: inputs[0].dataType }],
      dispatchGroup: { x: Math.ceil(outputSize / 64 /* workgroup size */) },
      programUniforms,
    }),
    getShaderSource,
  };
};

export const parseScatterNDAttributes = (attributes: Record<string, unknown>): ScatterNDAttributes =>
  createAttributeWithCacheKey({ reduction: attributes.reduction as string });

export const scatterND = (context: ComputeContext, attributes: ScatterNDAttributes): void => {
  context.compute(createScatterNDProgramInfo(context.inputs, attributes), {
    inputs: [context.inputs[1], context.inputs[2]],
    outputs: [],
  });
};
