// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from '../types';

import {inputVariable, outputVariable, ShaderHelper, tensorTypeToWsglStorageType} from './common';

export interface InstanceNormAttributes extends AttributeWithCacheKey {
  epsilon: number;
  format: 'NHWC'|'NCHW';
}

const createInstanceNormProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: InstanceNormAttributes): ProgramInfo => {
      const xShape = inputs[0].dims;

      const outputShape = xShape;
      const axis = 2;
      const normCount = ShapeUtil.sizeToDimension(xShape, axis);
      const normSize = ShapeUtil.sizeFromDimension(xShape, axis);
      const C = xShape[1];
      const x = inputVariable('x', inputs[0].dataType, [xShape[0], xShape[1], normSize]);
      const scale = inputVariable('scale', inputs[1].dataType, inputs[1].dims);
      const bias = inputVariable('bias', inputs[2].dataType, inputs[2].dims);
      const output = outputVariable('output', inputs[0].dataType, [xShape[0], xShape[1], normSize]);
      const variables = [x, scale, bias, output];
      const dataType = x.type.value;
      const workgroupSize = 64;
      const getShaderSource = (shaderHelper: ShaderHelper) => `

  const C: u32 = ${C};
  const normSize: u32 = ${normSize};
  const epsilon: f32 = ${attributes.epsilon};
  var<workgroup> meanShared : ${dataType};
  var<workgroup> squaredNormShared : ${dataType};
  var<workgroup> workgroupShared : array<${dataType}, ${workgroupSize}>;
  const workgroupSize = ${workgroupSize}u;
  ${shaderHelper.declareVariables(...variables)}
  ${shaderHelper.mainStart(workgroupSize)}
    let norm = global_idx / workgroupSize;
    let batch = norm / C;
    let channel = norm % C;
    let localIndex = local_id.x;

    // initialize workgroup memory
    var initial: ${dataType} = 0;
    for (var h = localIndex; h < normSize; h += workgroupSize) {
      initial = initial + ${x.get('batch', 'channel', 'h')};
    }
    workgroupShared[localIndex] = initial;
    workgroupBarrier();

    // Calculate the mean of current channel data.
    for (var currSize = workgroupSize >> 1;  currSize > 0; currSize = currSize >> 1) {
      if (localIndex < currSize) {
        workgroupShared[localIndex] = workgroupShared[localIndex] + workgroupShared[localIndex + currSize];
      }
      workgroupBarrier();
    }
    if (localIndex == 0) {
      meanShared = workgroupShared[0] / ${dataType}(normSize);
    }
    workgroupBarrier();

    // reinitialize workgroup memory.
    initial = 0;
    for (var h = localIndex; h < normSize; h += workgroupSize) {
      let deviation =  ${x.get('batch', 'channel', 'h')} - meanShared;
      initial = initial + deviation * deviation;
    }
    workgroupShared[localIndex] = initial;
    workgroupBarrier();

    // Calculate the sum of square of deviation of current channel data.
    for (var currSize = workgroupSize >> 1;  currSize > 0; currSize = currSize >> 1) {
      if (localIndex < currSize) {
        workgroupShared[localIndex] = workgroupShared[localIndex] + workgroupShared[localIndex + currSize];
      }
      workgroupBarrier();
    }
    if (localIndex == 0) {
      squaredNormShared = workgroupShared[0];
    }
    workgroupBarrier();

    let invStdDev = 1 / sqrt(squaredNormShared / ${dataType}(normSize) + epsilon);
    let channelScale = invStdDev * ${scale.getByOffset('channel')};
    let channelShift = ${bias.getByOffset('channel')} - meanShared * channelScale;
    for (var h = localIndex; h < normSize; h += workgroupSize) {
      let value = ${x.get('batch', 'channel', 'h')} * channelScale + channelShift;
      ${output.set('batch', 'channel', 'h', 'value')};
    }
  }`;
      return {
        ...metadata,
        outputs: [
          {dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default},
        ],
        getShaderSource,
        dispatchGroup: () => ({x: normCount})
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
