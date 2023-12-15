// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo} from '../types';

import {fillVector, getMaxComponents, inputVariable, outputVariable, ShaderHelper, sumVector, tensorTypeToWsglStorageType} from './common';

export interface InstanceNormAttributes extends AttributeWithCacheKey {
  epsilon: number;
  format: 'NHWC'|'NCHW';
}

const metadata = {
  name: 'InstanceNormalization'
};

const createInstanceNormProgramInfo =
    (inputs: readonly TensorView[], attributes: InstanceNormAttributes): ProgramInfo => {
      const xShape = inputs[0].dims;

      const outputShape = xShape;
      const axis = 2;
      const normCount = ShapeUtil.sizeToDimension(xShape, axis);
      const normSize = ShapeUtil.sizeFromDimension(xShape, axis);
      const components = getMaxComponents(normSize);
      const normPackedSize = normSize / components;
      const C = xShape[1];
      const x = inputVariable('x', inputs[0].dataType, [xShape[0], xShape[1], normPackedSize], components);
      const scale = inputVariable('scale', inputs[1].dataType, inputs[1].dims);
      const bias = inputVariable('bias', inputs[2].dataType, inputs[2].dims);
      const output = outputVariable('output', inputs[0].dataType, [xShape[0], xShape[1], normPackedSize], components);
      const variables = [x, scale, bias, output];
      const dataType = x.type.value;
      const f32Type = components === 1 ? 'f32' : `vec${components}<f32>`;
      const workgroupSize = 64;
      const getShaderSource = (shaderHelper: ShaderHelper) => `

  const C: u32 = ${C};
  const normSize: u32 = ${normSize};
  const epsilon: f32 = ${attributes.epsilon};
  var<workgroup> meanShared : f32;
  var<workgroup> squaredNormShared : f32;
  var<workgroup> workgroupShared : array<${f32Type}, ${workgroupSize}>;
  const workgroupSize = ${workgroupSize}u;
  ${shaderHelper.declareVariables(...variables)}
  ${shaderHelper.mainStart(workgroupSize)}
    let norm = global_idx / workgroupSize;
    let batch = norm / C;
    let channel = norm % C;
    let localIndex = local_id.x;

    // initialize workgroup memory
    var initial = ${f32Type}(0);
    for (var h = localIndex; h < ${normPackedSize}; h += workgroupSize) {
      initial = initial + ${f32Type}(${x.get('batch', 'channel', 'h')});
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
      meanShared = ${sumVector('workgroupShared[0]', components)} / f32(normSize);
    }
    workgroupBarrier();

    // reinitialize workgroup memory.
    initial = ${f32Type}(0);
    for (var h = localIndex; h < ${normPackedSize}; h += workgroupSize) {
      let deviation =  ${f32Type}(${x.get('batch', 'channel', 'h')}) - ${f32Type}(meanShared);
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
      squaredNormShared = ${sumVector('workgroupShared[0]', components)};
    }
    workgroupBarrier();

    let invStdDev = 1 / sqrt(squaredNormShared / f32(normSize) + epsilon);
    let channelScale = invStdDev * f32(${scale.getByOffset('channel')});
    let channelShift = f32(${bias.getByOffset('channel')}) - meanShared * channelScale;
    for (var h = localIndex; h < ${normPackedSize}; h += workgroupSize) {
      let value = ${x.get('batch', 'channel', 'h')} * ${dataType}(${f32Type}(channelScale)) + ${dataType}(${
          f32Type}(channelShift));
      ${output.set('batch', 'channel', 'h', 'value')};
    }
  }`;
      return {
        ...metadata,
        shaderCache: {hint: attributes.cacheKey},
        getRunData: () => ({
          outputs: [
            {dims: outputShape, dataType: inputs[0].dataType},
          ],
          dispatchGroup: {x: normCount}
        }),
        getShaderSource,
      };
    };

const computeMean =
    (context: ComputeContext, input: TensorView, scale: TensorView, bias: TensorView, n: number, h: number, c: number,
     epsilon: number) => {
      const components = getMaxComponents(c);
      const inputHelper = inputVariable('input', input.dataType, input.dims, components);
      const scaleHelper = inputVariable('scale', scale.dataType, scale.dims, components);
      const biasHelper = inputVariable('bias', bias.dataType, bias.dims, components);

      const WG = 64;
      // we will store channel scale and channel shift in [2, components] matrix
      // or in vec2 when components == 1
      const outputType = components === 1 ? 'vec2f' : `mat2x${components}f`;
      const sumCastType = components === 1 ? 'f32' : `vec${components}f`;
      const setOutputValue = (var1: string, var2: string) => `${outputType}(${var1}, ${var2})`;
      const unitsOfWork = n * c / components;
      const wgSize = Math.ceil(h / WG);

      const getMeanShaderSource = (shaderHelper: ShaderHelper) => `
  const H: u32 = ${h};
  const C: u32 = ${c / components};
  const imageSize: u32 = ${h * c / components};

  ${shaderHelper.declareVariables(inputHelper)}
  @group(0) @binding(1) var<storage, read_write> output : array<${outputType}>;

  ${shaderHelper.mainStart(WG)}
    let currentImageNumber = global_idx / ${WG} / C;
    let currentChannelNumber = (global_idx / ${WG}) % C;
    let wgId = global_idx % ${WG};
    let wgOffset = wgId * ${wgSize};
    if (wgOffset >= H) {
        return;
    }
    let wgMax = min(wgOffset + ${wgSize}, H);

    let offset = currentImageNumber * imageSize + currentChannelNumber;
    var sum = ${fillVector('f32', components)};
    var squaredSum = ${fillVector('f32', components)};
    for (var i: u32 = wgOffset; i < wgMax; i++) {
        let value = ${sumCastType}(input[offset + i * C]);
        sum += value;
        squaredSum += value * value;
    }
    output[global_idx] = ${setOutputValue('sum', 'squaredSum')};
  }`;

      const meanValues = context.compute(
          {
            name: 'InstanceNormComputeMean',
            shaderCache: {hint: JSON.stringify({components, n, h, c})},
            getRunData: () => ({
              outputs: [
                {dims: [n, c, WG, 2], dataType: DataType.float},
              ],
              dispatchGroup: {x: n * c / components},
            }),
            getShaderSource: getMeanShaderSource,
          },
          {inputs: [input], outputs: [-1]})[0];
      const getShaderSource = (shaderHelper: ShaderHelper) => `
  const H: u32 = ${h};
  const C: u32 = ${c / components};
  const imageSize: u32 = ${WG * c / components};
  const epsilon: f32 = ${epsilon};

  @group(0) @binding(0) var<storage, read> input : array<${outputType}>;
  @group(0) @binding(1) var<storage, read> scale : array<${scaleHelper.type.storage}>;
  @group(0) @binding(2) var<storage, read> bias : array<${biasHelper.type.storage}>;
  @group(0) @binding(3) var<storage, read_write> output : array<${outputType}>;

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(unitsOfWork)}
    let currentImageNumber = global_idx / C;
    let currentChannelNumber = global_idx % C;

    let offset = currentImageNumber * imageSize;
    var sum = ${fillVector('f32', components)};
    var squaredSum = ${fillVector('f32', components)};
    for (var i: u32 = 0; i < ${WG}; i++) {
        let value = input[offset + i + currentChannelNumber * ${WG}];
        sum += value[0];
        squaredSum += value[1];
    }
    sum = sum / f32(H);
    squaredSum = squaredSum / f32(H);
    let invStdDev = 1 / sqrt(squaredSum - sum * sum + epsilon);
    let channelScale = invStdDev * ${sumCastType}(scale[currentChannelNumber]);
    let channelShift = ${sumCastType}(bias[currentChannelNumber]) - sum * channelScale;

    output[global_idx] = ${setOutputValue('channelScale', 'channelShift')};
  }`;

      return context.compute(
          {
            name: 'InstanceNormComputeChannelScaleShift',
            shaderCache: {hint: JSON.stringify({components, n, h, c, epsilon})},
            getRunData: () => ({
              outputs: [
                {dims: [n, c, 2], dataType: DataType.float},
              ],
              dispatchGroup: {x: Math.ceil(unitsOfWork / 64 /* workgroup size */)},
            }),
            getShaderSource,
          },
          {inputs: [meanValues, scale, bias], outputs: [-1]})[0];
    };

const createInstanceNormNHWCProgramInfo =
    (context: ComputeContext, inputs: readonly TensorView[], attributes: InstanceNormAttributes) => {
      const xShape = inputs[0].dims;
      const outputShape = xShape;
      const N = xShape[0];
      const C = xShape[xShape.length - 1];
      const H = ShapeUtil.sizeFromDimension(xShape, 1) / C;

      const components = getMaxComponents(C);
      const outputSize = ShapeUtil.size(outputShape) / components;
      const inputHelper = inputVariable('input', inputs[0].dataType, inputs[0].dims, components);
      const outputHelper = outputVariable('output', inputs[0].dataType, outputShape, components);

      const dataType = tensorTypeToWsglStorageType(inputs[0].dataType);
      const scaleType = components === 1 ? 'vec2f' : `mat2x${components}f`;
      const scaleCastType = components === 1 ? dataType : `vec${components}<${dataType}>`;
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
    output[global_idx] = fma(input[global_idx], ${scaleCastType}(scale[0]), ${scaleCastType}(scale[1]));
  }`;
      context.compute(
          {
            name: 'InstanceNormalization',
            shaderCache: {hint: `${attributes.cacheKey}`},
            getRunData: () => ({
              outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
              dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)}
            }),
            getShaderSource,
          },
          {inputs: [inputs[0], channelScaleShift]});
    };

export const parseInstanceNormAttributes = (attributes: InstanceNormAttributes): InstanceNormAttributes =>
    createAttributeWithCacheKey({epsilon: attributes.epsilon, format: attributes.format});

export const instanceNorm = (context: ComputeContext, attributes: InstanceNormAttributes): void => {
  if (attributes.format === 'NHWC') {
    createInstanceNormNHWCProgramInfo(context, context.inputs, attributes);
  } else {
    context.compute(createInstanceNormProgramInfo(context.inputs, attributes));
  }
};
