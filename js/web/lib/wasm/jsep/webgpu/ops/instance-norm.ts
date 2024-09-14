// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { DataType } from '../../../wasm-common';
import { TensorView } from '../../tensor-view';
import { ShapeUtil } from '../../util';
import { ComputeContext, ProgramInfo, ProgramInputTensorInfoDependency, ProgramUniform } from '../types';
import { createTransposeProgramInfo } from './transpose';

import {
  createTensorShapeVariables,
  getMaxComponents,
  inputVariable,
  outputVariable,
  ShaderHelper,
  sumVector,
  tensorTypeToWsglStorageType,
} from './common';

export interface InstanceNormAttributes {
  epsilon: number;
  format: 'NHWC' | 'NCHW';
}

const createInstanceNormProgramInfo = (
  inputs: readonly TensorView[],
  attributes: InstanceNormAttributes,
): ProgramInfo => {
  const xShape = inputs[0].dims;
  const outputShape = xShape;
  const axis = 2;
  const normCount = ShapeUtil.sizeToDimension(xShape, axis);
  const normSize = ShapeUtil.sizeFromDimension(xShape, axis);
  const components = getMaxComponents(normSize);
  const normPackedSize = normSize / components;
  const inputShape = [xShape[0], xShape[1], normPackedSize];
  const inputDependencies: ProgramInputTensorInfoDependency[] = ['rank', 'type', 'type'];
  const programUniforms: ProgramUniform[] = [];
  programUniforms.push(...createTensorShapeVariables(inputShape, inputShape));

  const getShaderSource = (shaderHelper: ShaderHelper) => {
    const x = inputVariable('x', inputs[0].dataType, inputShape.length, components);
    const scale = inputVariable('scale', inputs[1].dataType, inputs[1].dims);
    const bias = inputVariable('bias', inputs[2].dataType, inputs[2].dims);
    const output = outputVariable('output', inputs[0].dataType, inputShape.length, components);
    const variables = [x, scale, bias, output];
    const dataType = x.type.value;
    const f32Type = components === 1 ? 'f32' : `vec${components}<f32>`;
    const wgType = components === 1 ? 'vec2f' : `mat2x${components}f`;
    const workgroupSize = 64;

    return `
  var<workgroup> channel_scale : f32;
  var<workgroup> channel_shift : f32;
  var<workgroup> workgroup_shared : array<${wgType}, ${workgroupSize}>;
  const workgroup_size = ${workgroupSize}u;
  ${shaderHelper.declareVariables(...variables)}
  ${shaderHelper.mainStart(workgroupSize)}
    let batch = workgroup_index / uniforms.x_shape[1];
    let channel = workgroup_index % uniforms.x_shape[1];
    let hight = uniforms.x_shape[2];
    // initialize workgroup memory
    var sum = ${f32Type}(0);
    var squared_sum = ${f32Type}(0);
    for (var h = local_idx; h < hight; h += workgroup_size) {
      let value = ${f32Type}(${x.get('batch', 'channel', 'h')});
      sum += value;
      squared_sum += value * value;
    }
    workgroup_shared[local_idx] = ${wgType}(sum, squared_sum);
    workgroupBarrier();

    // Calculate the mean of current channel data.
    for (var currSize = workgroup_size >> 1;  currSize > 0; currSize = currSize >> 1) {
      if (local_idx < currSize) {
        workgroup_shared[local_idx] = workgroup_shared[local_idx] + workgroup_shared[local_idx + currSize];
      }
      workgroupBarrier();
    }
    if (local_idx == 0) {
      let sum_final = ${sumVector('workgroup_shared[0][0]', components)} / f32(hight * ${components});
      let squared_sum_final = ${sumVector('workgroup_shared[0][1]', components)} / f32(hight * ${components});

      let inv_std_dev = inverseSqrt(squared_sum_final - sum_final * sum_final + f32(${attributes.epsilon}));
      channel_scale = inv_std_dev * f32(scale[channel]);
      channel_shift = f32(bias[channel]) - sum_final * channel_scale;
    }
    workgroupBarrier();

    for (var h = local_idx; h < hight; h += workgroup_size) {
      let value = ${x.get('batch', 'channel', 'h')} * ${dataType}(${f32Type}(channel_scale)) + ${dataType}(${
        f32Type
      }(channel_shift));
      ${output.set('batch', 'channel', 'h', 'value')};
    }
  }`;
  };
  return {
    ...{ name: 'InstanceNormalization' },
    // TODO: use epsilon as uniform. Currently epsilon as uniform fails test_instancenorm_epsilon.
    shaderCache: { hint: `${attributes.epsilon};${components}`, inputDependencies },
    getRunData: () => ({
      outputs: [{ dims: outputShape, dataType: inputs[0].dataType }],
      dispatchGroup: { x: normCount },
      programUniforms,
    }),
    getShaderSource,
  };
};

const computeChannelScaleShift = (
  context: ComputeContext,
  input: TensorView,
  scale: TensorView,
  bias: TensorView,
  n: number,
  h: number,
  c: number,
  epsilon: number,
) => {
  // transpose x from NHWC to NCHW
  const xShape = context.inputs[0].dims;
  const transposedXPerm = [0, xShape.length - 1];
  for (let i = 0; i < xShape.length - 2; i++) {
    transposedXPerm.push(i + 1);
  }
  const transposedX = context.compute(createTransposeProgramInfo(input, transposedXPerm), {
    inputs: [context.inputs[0]],
    outputs: [-1],
  })[0];

  const components = getMaxComponents(h);
  const f32Type = components === 1 ? 'f32' : `vec${components}f`;
  const wgType = components === 1 ? 'vec2f' : `mat2x${components}f`;
  const unitsOfWork = n * c;

  const inputShape = [n, c, h / components];
  const outputShape = [n, c, 2];
  const inputDependencies: ProgramInputTensorInfoDependency[] = ['rank', 'type', 'type'];
  const programUniforms: ProgramUniform[] = [];
  programUniforms.push(...createTensorShapeVariables(inputShape, outputShape));

  const getShaderSource = (shaderHelper: ShaderHelper) => {
    const x = inputVariable('x', input.dataType, 3, components);
    const s = inputVariable('scale', scale.dataType, scale.dims);
    const b = inputVariable('bias', bias.dataType, bias.dims);
    const output = outputVariable('output', DataType.float, 3, 2);
    const variables = [x, s, b, output];
    const workgroupSize = 64;
    return `
  var<workgroup> workgroup_shared : array<${wgType}, ${workgroupSize}>;
  const workgroup_size = ${workgroupSize}u;
  ${shaderHelper.declareVariables(...variables)}
  ${shaderHelper.mainStart(workgroupSize)}
    let batch = workgroup_index / uniforms.x_shape[1];
    let channel = workgroup_index % uniforms.x_shape[1];
    let hight = uniforms.x_shape[2];
    // initialize workgroup memory
    var sum = ${f32Type}(0);
    var squared_sum = ${f32Type}(0);
    for (var h = local_idx; h < hight; h += workgroup_size) {
      let value = ${f32Type}(${x.get('batch', 'channel', 'h')});
      sum += value;
      squared_sum += value * value;
    }
    workgroup_shared[local_idx] = ${wgType}(sum, squared_sum);
    workgroupBarrier();

    for (var currSize = workgroup_size >> 1;  currSize > 0; currSize = currSize >> 1) {
      if (local_idx < currSize) {
        workgroup_shared[local_idx] = workgroup_shared[local_idx] + workgroup_shared[local_idx + currSize];
      }
      workgroupBarrier();
    }
    if (local_idx == 0) {
      let sum_final = ${sumVector('workgroup_shared[0][0]', components)} / f32(hight * ${components});
      let squared_sum_final = ${sumVector('workgroup_shared[0][1]', components)} / f32(hight * ${components});

      let inv_std_dev = inverseSqrt(squared_sum_final - sum_final * sum_final + f32(${epsilon}));
      let channel_scale = inv_std_dev * f32(scale[channel]);
      let channel_shift = f32(bias[channel]) - sum_final * channel_scale;
      output[workgroup_index] = vec2f(channel_scale, channel_shift);
    }
  }`;
  };

  return context.compute(
    {
      name: 'InstanceNormComputeChannelScaleShift',
      // TODO: use epsilon as uniform. Currently epsilon as uniform fails test_instancenorm_epsilon.
      shaderCache: { hint: `${components};${epsilon}`, inputDependencies },
      getRunData: () => ({
        outputs: [{ dims: outputShape, dataType: DataType.float }],
        dispatchGroup: { x: unitsOfWork },
        programUniforms,
      }),
      getShaderSource,
    },
    { inputs: [transposedX, scale, bias], outputs: [-1] },
  )[0];
};

const createInstanceNormNHWCProgramInfo = (
  context: ComputeContext,
  inputs: readonly TensorView[],
  attributes: InstanceNormAttributes,
) => {
  const xShape = inputs[0].dims;
  const outputShape = xShape;
  const N = xShape[0];
  const C = xShape[xShape.length - 1];
  const H = ShapeUtil.sizeFromDimension(xShape, 1) / C;
  const components = getMaxComponents(C);
  const outputSize = ShapeUtil.size(outputShape) / components;
  const programUniforms: ProgramUniform[] = [
    { type: DataType.uint32, data: H },
    { type: DataType.uint32, data: Math.floor(C / components) },
  ];
  const inputDependencies: ProgramInputTensorInfoDependency[] = ['type', 'type'];
  // first compute mean
  const channelScaleShift = computeChannelScaleShift(
    context,
    inputs[0],
    inputs[1],
    inputs[2],
    N,
    H,
    C,
    attributes.epsilon,
  );
  const getShaderSource = (shaderHelper: ShaderHelper) => {
    const dataType = tensorTypeToWsglStorageType(inputs[0].dataType);
    const scaleType = components === 1 ? 'vec2f' : `mat${components}x2f`;
    const scaleData = (num: number) => {
      const index = num === 0 ? 'x' : 'y';
      const f32Type = components === 1 ? 'f32' : `vec${components}f`;
      switch (components) {
        case 1:
          return `${dataType}(${f32Type}(scale.${index}))`;
        case 2:
          return `vec2<${dataType}>(${f32Type}(scale[0].${index}, scale[1].${index}))`;
        case 4:
          return `vec4<${dataType}>(${f32Type}(scale[0].${index}, scale[1].${index}, scale[2].${index}, scale[3].${index}))`;
        default:
          throw new Error(`Not supported compoents ${components}`);
      }
    };
    const inputHelper = inputVariable('input', inputs[0].dataType, inputs[0].dims, components);
    const outputHelper = outputVariable('output', inputs[0].dataType, outputShape, components);

    return `
  @group(0) @binding(0) var<storage, read> input : array<${inputHelper.type.storage}>;
  @group(0) @binding(1) var<storage, read> scale_input : array<${scaleType}>;
  @group(0) @binding(2) var<storage, read_write> output : array<${outputHelper.type.storage}>;
  struct Uniforms {H: u32, C : u32};
  @group(0) @binding(3) var<uniform> uniforms: Uniforms;

  ${shaderHelper.mainStart()}
    let current_image_number = global_idx / (uniforms.C * uniforms.H);
    let current_channel_number = global_idx % uniforms.C;

    let scale_offset = current_image_number * uniforms.C + current_channel_number;
    let scale = scale_input[scale_offset];
    output[global_idx] = fma(input[global_idx], ${scaleData(0)}, ${scaleData(1)});
  }`;
  };
  context.compute(
    {
      name: 'InstanceNormalizationNHWC',
      shaderCache: { hint: `${components}`, inputDependencies },
      getRunData: () => ({
        outputs: [{ dims: outputShape, dataType: inputs[0].dataType }],
        dispatchGroup: { x: Math.ceil(outputSize / 64 /* workgroup size */) },
        programUniforms,
      }),
      getShaderSource,
    },
    { inputs: [inputs[0], channelScaleShift] },
  );
};

export const instanceNorm = (context: ComputeContext, attributes: InstanceNormAttributes): void => {
  if (attributes.format === 'NHWC') {
    createInstanceNormNHWCProgramInfo(context, context.inputs, attributes);
  } else {
    context.compute(createInstanceNormProgramInfo(context.inputs, attributes));
  }
};
