// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from '../types';

import {inputVariable, outputVariable, ShaderHelper} from './common';
import {createTransposeProgramInfo, TransposeAttributes, transposeProgramMetadata} from './transpose';

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

export const parseInstanceNormAttributes = (attributes: InstanceNormAttributes): InstanceNormAttributes =>
    createAttributeWithCacheKey({epsilon: attributes.epsilon, format: attributes.format});

export const instanceNorm = (context: ComputeContext, attributes: InstanceNormAttributes): void => {
  const metadata = {
    name: 'InstanceNormalization',
    inputTypes: [GpuDataType.default, GpuDataType.default, GpuDataType.default],
    cacheHint: attributes.cacheKey,
  };

  if (attributes.format === 'NHWC') {
    // transpose x from NHWC to NCHW
    const xShape = context.inputs[0].dims;
    const transposedXPerm = [0, xShape.length - 1];
    for (let i = 0; i < xShape.length - 2; i++) {
      transposedXPerm.push(i + 1);
    }
    const xTransposeAttribute: TransposeAttributes = createAttributeWithCacheKey({perm: transposedXPerm});
    const transposedX = context.compute(
        {
          ...transposeProgramMetadata,
          cacheHint: xTransposeAttribute.cacheKey,
          get: () => createTransposeProgramInfo(context.inputs[0], xTransposeAttribute.perm)
        },
        {inputs: [context.inputs[0]], outputs: [-1]})[0];
    const inputs = [transposedX, context.inputs[1], context.inputs[2]];
    const y = context.compute(createInstanceNormProgramInfo(metadata, inputs, attributes), {inputs, outputs: [-1]})[0];
    // transpose y from NCHW to NHWC again.
    const transposedYPerm = [0];
    for (let i = 0; i < xShape.length - 2; i++) {
      transposedYPerm.push(i + 2);
    }
    transposedYPerm.push(1);
    const yTransposeAttribute: TransposeAttributes = createAttributeWithCacheKey({perm: transposedYPerm});
    context.compute(
        {
          ...transposeProgramMetadata,
          cacheHint: yTransposeAttribute.cacheKey,
          get: () => createTransposeProgramInfo(y, yTransposeAttribute.perm)
        },
        {inputs: [y]});
  } else {
    context.compute(createInstanceNormProgramInfo(metadata, context.inputs, attributes));
  }
};
