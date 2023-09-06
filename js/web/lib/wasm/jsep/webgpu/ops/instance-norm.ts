// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from '../types';

import {
  ShaderHelper,
  inputVariable,
  tensorTypeToWsglStorageType,
  outputVariable,
  getMaxComponents,
  fillVector
} from './common'
import { DataType } from '../../../wasm-common'

export interface InstanceNormAttributes extends AttributeWithCacheKey {
  epsilon: number;
  format: 'NHWC'|'NCHW';
}

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length !== 3) {
    throw new Error('instanceNorm requires 3 inputs.');
  }
};

const createInstanceNormProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: InstanceNormAttributes): ProgramInfo => {
      const xShape = inputs[0].dims;
      const outputShape = xShape;
      const outputSize = ShapeUtil.size(outputShape);
      const axis = 2;
      const normCount = xShape[0] * xShape[1];
      const normSize = ShapeUtil.sizeFromDimension(xShape, axis);
      const C = xShape[1];

      const dataType = tensorTypeToWsglStorageType(inputs[0].dataType);

      const getShaderSource = (shaderHelper: ShaderHelper) => `
  const C: u32 = ${C};
  const normSize: u32 = ${normSize};
  const normSizeTyped: ${dataType} = ${normSize};
  const epsilon: f32 = ${attributes.epsilon};

  @group(0) @binding(0) var<storage, read> x : array<${dataType}>;
  @group(0) @binding(1) var<storage, read> scale : array<${dataType}>;
  @group(0) @binding(2) var<storage, read> bias : array<${dataType}>;
  @group(0) @binding(3) var<storage, read_write> output : array<${dataType}>;

  ${shaderHelper.mainStart()}
    let offset = global_idx * normSize;
    if (offset >= ${outputSize}) { return; }
    var mean: ${dataType} = 0;

    for (var h: u32 = 0u; h < normSize; h++) {
        mean = mean + x[h + offset];
    }
    mean = mean / normSizeTyped;

    var squaredNorm: ${dataType} = 0;
    for (var h: u32 = 0u; h < normSize; h++) {
        let deviation: f32 = x[h + offset] - mean;
        squaredNorm = squaredNorm + deviation * deviation;
    }
    let invStdDev = 1 / sqrt(squaredNorm / normSizeTyped + epsilon);
    let channelScale = invStdDev * scale[global_idx % C];
    let channelShift = bias[global_idx % C] - mean * channelScale;
    for (var j: u32 = 0; j < normSize; j++) {
        output[j + offset] = x[j + offset] * channelScale + channelShift;
    }
  }`;
      return {
        ...metadata,
        outputs: [
          {dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default},
        ],
        getShaderSource,
        dispatchGroup: () => ({x: Math.ceil(normCount / 64 /* workgroup size */)})
      };
    };

const computeMean = (context: ComputeContext, input: TensorView, scale: TensorView, bias: TensorView, n: number, h: number, c: number, epsilon: number) => {
  const components = getMaxComponents(c);
  const inputHelper = inputVariable('input', input.dataType, input.dims, components);
  const scaleHelper = inputVariable('scale', scale.dataType, scale.dims, components);
  const biasHelper = inputVariable('bias', bias.dataType, bias.dims, components);

  const WG = 64;
  // we will store channel scale and channel shift in [2, components] matrix
  // or in vec2 when components == 1
  const outputType = components === 1 ? 'array<vec2<f32>>' : `array<mat2x${components}f>`;
  const setOutputValue = (var1: string, var2: string) => {
    return components === 1
      ? `vec2f(${var1}, ${var2})`
      : `mat2x${components}<f32>(${var1}, ${var2})`;
  };
  const unitsOfWork = n * c / components;
  const wgSize = Math.ceil(h / WG);
  const getMeanShaderSource = (shaderHelper: ShaderHelper) => `
  const H: u32 = ${h};
  const C: u32 = ${c / components};
  const imageSize: u32 = ${h * c / components};

  ${shaderHelper.declareVariables(inputHelper)}
  @group(0) @binding(1) var<storage, read_write> output : ${outputType};

  ${shaderHelper.mainStart(64)}
    let currentImageNumber = global_idx / ${WG} / C;
    let currentChannelNumber = (global_idx / ${WG}) % C;
    let wgId = global_idx % 64;
    let wgOffset = wgId * ${wgSize};
    if (wgOffset > H) {
        return;
    }
    let wgMax = min(wgOffset + ${wgSize}, H);

    let offset = currentImageNumber * imageSize + currentChannelNumber;
    var sum: ${inputHelper.type.storage} = ${fillVector('f32', components)};
    var squaredSum: ${inputHelper.type.storage} = ${fillVector('f32', components)};
    for (var i: u32 = wgOffset; i < wgMax; i++) {
        let value = input[offset + i * C];
        sum += value;
        squaredSum += value * value;
    }
    output[global_idx] = ${setOutputValue('sum', 'squaredSum')};
  }`;

  const meanValues = context.compute(
    {
      name: 'InstanceNormComputeMean',
      inputTypes: [GpuDataType.default],
      cacheHint: JSON.stringify({ components, n, h, c }),
      outputs: [
        {dims: [n, c, WG, 2], dataType: DataType.float, gpuDataType: GpuDataType.default},
      ],
      getShaderSource: getMeanShaderSource,
      dispatchGroup: () => ({x: n * c / components})
    },
    {inputs: [input], outputs: [-1]})[0];
  const getShaderSource = (shaderHelper: ShaderHelper) => `
  const H: u32 = ${h};
  const C: u32 = ${c / components};
  const imageSize: u32 = ${WG * c / components};
  const epsilon: f32 = ${epsilon};

  @group(0) @binding(0) var<storage, read> input : ${outputType};
  @group(0) @binding(1) var<storage, read> scale : array<${scaleHelper.type.storage}>;
  @group(0) @binding(2) var<storage, read> bias : array<${biasHelper.type.storage}>;
  @group(0) @binding(3) var<storage, read_write> output : ${outputType};

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(unitsOfWork)}
    let currentImageNumber = global_idx / C;
    let currentChannelNumber = global_idx % C;

    let offset = currentImageNumber * imageSize;
    var sum: ${inputHelper.type.storage} = ${fillVector('f32', components)};
    var squaredSum: ${inputHelper.type.storage} = ${fillVector('f32', components)};
    for (var i: u32 = 0; i < ${WG}; i++) {
        let value = input[offset + i];
        sum += value[0];
        squaredSum += value[1];
    }
    let mean = sum / f32(H);
    let invStdDev = 1 / sqrt(squaredSum / f32(H) - mean * mean + epsilon);
    let channelScale = invStdDev * scale[currentChannelNumber];
    let channelShift = bias[currentChannelNumber] - mean * channelScale;

    output[global_idx] = ${setOutputValue('channelScale', 'channelShift')};
  }`;

  return context.compute(
    {
      name: 'InstanceNormComputeMean',
      inputTypes: [GpuDataType.default, GpuDataType.default, GpuDataType.default],
      cacheHint: JSON.stringify({ components, n, h, c, epsilon }),
      outputs: [
        {dims: [n, c, 2], dataType: DataType.float, gpuDataType: GpuDataType.default},
        // {dims: [h * c], dataType: DataType.float, gpuDataType: GpuDataType.default},
      ],
      getShaderSource,
      dispatchGroup: () => ({x: Math.ceil(unitsOfWork / 64 /* workgroup size */)})
    },
    {inputs: [meanValues, scale, bias], outputs: [-1]})[0];
};

const createInstanceNormNHWCProgramInfo =
    (context: ComputeContext, metadata: ProgramMetadata, inputs: readonly TensorView[],
      attributes: InstanceNormAttributes) => {
      const xShape = inputs[0].dims;
      const outputShape = xShape;
      const outputSize = ShapeUtil.size(outputShape);
      const N = xShape[0];
      const C = xShape[xShape.length - 1];
      const H = ShapeUtil.sizeFromDimension(xShape, 1) / C;

      const components = getMaxComponents(C);
      const inputHelper = inputVariable('input', inputs[0].dataType, inputs[0].dims, components);
      const outputHelper = outputVariable('output', inputs[0].dataType, outputShape, components);

      const scaleType = components === 1 ? 'vec2<f32>' : `mat2x${components}f`;
      // first compute mean
      const channelScaleShift = computeMean(context, inputs[0], inputs[1], inputs[2], N, H, C, attributes.epsilon);

      const getShaderSource = (shaderHelper: ShaderHelper) => `
  const H: u32 = ${H};
  const C: u32 = ${C / components};

  @group(0) @binding(0) var<storage, read> input : array<${inputHelper.type.storage}>;
  @group(0) @binding(1) var<storage, read> scaleInput : array<${scaleType}>;
  @group(0) @binding(2) var<storage, read_write> output : array<${outputHelper.type.storage}>;

  ${shaderHelper.mainStart()}
    let currentImageNumber = global_idx / (C * H);
    let currentChannelNumber = global_idx % C;

    let scaleOffset = currentImageNumber * C + currentChannelNumber;
    let scale = scaleInput[scaleOffset];
    output[global_idx] = fma(input[global_idx], scale[0], scale[1]);
  }`;
      context.compute({
        ...metadata,
        inputTypes: [GpuDataType.default, GpuDataType.default],
        outputs: [
          {dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default},
        ],
        getShaderSource,
        dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
      },
      {
        inputs: [inputs[0], channelScaleShift]
      });
    };

export const parseInstanceNormAttributes = (attributes: InstanceNormAttributes): InstanceNormAttributes =>
    createAttributeWithCacheKey({epsilon: attributes.epsilon, format: attributes.format});

export const instanceNorm = (context: ComputeContext, attributes: InstanceNormAttributes): void => {
  validateInputs(context.inputs);

  const metadata = {
    name: 'InstanceNormalization',
    inputTypes: [GpuDataType.default, GpuDataType.default, GpuDataType.default],
    cacheHint: attributes.cacheKey,
  };

  if (attributes.format === 'NHWC') {
    createInstanceNormNHWCProgramInfo(context, metadata, context.inputs, attributes);
  } else {
    context.compute(createInstanceNormProgramInfo(metadata, context.inputs, attributes));
  }
};
