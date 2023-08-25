// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from '../types';

import {ShaderHelper, tensorTypeToWsglStorageType} from './common';

export interface InstanceNormAttributes extends AttributeWithCacheKey {
  epsilon: number;
  format: 'NHWC'|'NCHW';
}

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length !== 3) {
    throw new Error('instanceNorm requires 3 inputs.');
  }

  if (inputs[0].dataType !== DataType.float || inputs[1].dataType !== DataType.float) {
    throw new Error('inputs should be float type');
  }
};

const createInstanceNormProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: InstanceNormAttributes): ProgramInfo => {
      const xShape = inputs[0].dims;
      const scale = inputs[1];
      const bias = inputs[2];

      const outputShape = xShape;
      const outputSize = ShapeUtil.size(outputShape);
      const axis = 2;
      const normCount = ShapeUtil.sizeToDimension(xShape, axis);
      const normSize = ShapeUtil.sizeFromDimension(xShape, axis);
      const C = xShape[1];

      const scaleSize = ShapeUtil.size(scale.dims);
      const biasSize = bias ? ShapeUtil.size(bias.dims) : 0;
      if (scaleSize !== normSize || (bias && biasSize !== normSize)) {
        throw new Error(`Size of X.shape()[axis:] == ${normSize}.
             Size of scale and bias (if provided) must match this. 
             Got scale size of ${scaleSize} and bias size of ${biasSize}`);
      }

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
    if (offset + normSize >= ${outputSize}) { return; }
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

const createInstanceNormNHWCProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: InstanceNormAttributes): ProgramInfo => {
      const xShape = inputs[0].dims;
      const outputShape = xShape;
      const outputSize = ShapeUtil.size(outputShape);
      const N = xShape[0];
      const C = xShape[xShape.length - 1];
      const H = ShapeUtil.sizeFromDimension(xShape, 1) / C;

      const dataType = tensorTypeToWsglStorageType(inputs[0].dataType);

      const normCount = C * N;
      const getShaderSource = (shaderHelper: ShaderHelper) => `
  const N: u32 = ${N};
  const H: u32 = ${H};
  const C: u32 = ${C};
  const normSizeTyped: ${dataType} = ${H};
  const imageSize: u32 = ${H * C};
  const epsilon: f32 = ${attributes.epsilon};

  @group(0) @binding(0) var<storage, read> x : array<${dataType}>;
  @group(0) @binding(1) var<storage, read> scale : array<${dataType}>;
  @group(0) @binding(2) var<storage, read> bias : array<${dataType}>;
  @group(0) @binding(3) var<storage, read_write> output : array<${dataType}>;

  ${shaderHelper.mainStart()}
    let currentImageNumber = global_idx / C;
    let currentChannelNumber = global_idx % C;
    
    // offset is channel num * N
    let offset = currentImageNumber * imageSize;
    if (offset >= ${outputSize}) { return; }
    var mean: ${dataType} = 0;

    for (var i: u32 = 0u; i < H; i++) {
        mean = mean + x[offset + i * C + currentChannelNumber];
    }
    mean = mean / normSizeTyped;

    var squaredNorm: ${dataType} = 0;
    for (var i: u32 = 0u; i < H; i++) {
        let deviation: f32 = x[offset + i * C + currentChannelNumber] - mean;
        squaredNorm = squaredNorm + deviation * deviation;
    }
    let invStdDev = 1 / sqrt(squaredNorm / normSizeTyped + epsilon);
    let channelScale = invStdDev * scale[currentChannelNumber];
    let channelShift = bias[currentChannelNumber] - mean * channelScale;
    for (var i: u32 = 0u; i < H; i++) {
        let currentOffset = offset + i * C + currentChannelNumber;
        output[currentOffset] = x[currentOffset] * channelScale + channelShift;
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
    context.compute(createInstanceNormNHWCProgramInfo(metadata, context.inputs, attributes));
  } else {
    context.compute(createInstanceNormProgramInfo(metadata, context.inputs, attributes));
  }
};
