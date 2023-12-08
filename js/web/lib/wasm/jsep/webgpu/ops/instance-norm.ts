// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {ComputeContext, ProgramInfo, ProgramInputTensorInfoDependency, ProgramUniform} from '../types';

import {createTensorShapeVariables, fillVector, getMaxComponents, inputVariable, outputVariable, ShaderHelper, sumVector, tensorTypeToWsglStorageType, UniformsArrayType} from './common';

export interface InstanceNormAttributes {
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
      const inputShape = [xShape[0], xShape[1], normPackedSize];
      const inputDependencies: ProgramInputTensorInfoDependency[] = ['rank', 'rank', 'rank'];
      const programUniforms: ProgramUniform[] =
          [{type: 'uint32', data: C}, {type: 'uint32', data: normSize}, {type: 'uint32', data: normPackedSize}];
      programUniforms.push(
          ...createTensorShapeVariables(inputShape), ...createTensorShapeVariables(inputs[1].dims),
          ...createTensorShapeVariables(inputs[2].dims), ...createTensorShapeVariables(inputShape));

      const getShaderSource = (shaderHelper: ShaderHelper) => {
        const x = inputVariable('x', inputs[0].dataType, inputShape.length, components);
        const scale = inputVariable('scale', inputs[1].dataType, inputs[1].dims.length);
        const bias = inputVariable('bias', inputs[2].dataType, inputs[2].dims.length);
        const output = outputVariable('output', inputs[0].dataType, inputShape.length, components);
        const variables = [x, scale, bias, output];
        const dataType = x.type.value;
        const f32Type = components === 1 ? 'f32' : `vec${components}<f32>`;
        const workgroupSize = 64;

        const uniforms: UniformsArrayType =
            [{name: 'C', type: 'u32'}, {name: 'normSize', type: 'u32'}, {name: 'normPackedSize', type: 'u32'}];
        return `
  var<workgroup> meanShared : f32;
  var<workgroup> squaredNormShared : f32;
  var<workgroup> workgroupShared : array<${f32Type}, ${workgroupSize}>;
  const workgroupSize = ${workgroupSize}u;
  ${shaderHelper.registerUniforms(uniforms).declareVariables(...variables)}
  ${shaderHelper.mainStart(workgroupSize)}
    let norm = global_idx / workgroupSize;
    let batch = norm / uniforms.C;
    let channel = norm % uniforms.C;
    let localIndex = local_id.x;

    // initialize workgroup memory
    var initial = ${f32Type}(0);
    for (var h = localIndex; h < uniforms.normPackedSize; h += workgroupSize) {
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
      meanShared = ${sumVector('workgroupShared[0]', components)} / f32(uniforms.normSize);
    }
    workgroupBarrier();

    // reinitialize workgroup memory.
    initial = ${f32Type}(0);
    for (var h = localIndex; h < uniforms.normPackedSize; h += workgroupSize) {
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

    let invStdDev = 1 / sqrt(squaredNormShared / f32(uniforms.normSize) + f32(${attributes.epsilon}));
    let channelScale = invStdDev * f32(${scale.getByOffset('channel')});
    let channelShift = f32(${bias.getByOffset('channel')}) - meanShared * channelScale;
    for (var h = localIndex; h < uniforms.normPackedSize; h += workgroupSize) {
      let value = ${x.get('batch', 'channel', 'h')} * ${dataType}(${f32Type}(channelScale)) + ${dataType}(${
            f32Type}(channelShift));
      ${output.set('batch', 'channel', 'h', 'value')};
    }
  }`;
      };
      return {
        ...metadata,
        // TODO: use epsilon as uniform. Currently epsilon as uniform fails test_instancenorm_epsilon.
        shaderCache: {hint: `${attributes.epsilon}`, inputDependencies},
        getRunData: () => ({
          outputs: [
            {dims: outputShape, dataType: inputs[0].dataType},
          ],
          dispatchGroup: {x: normCount},
          programUniforms
        }),
        getShaderSource,
      };
    };

const computeMean =
    (context: ComputeContext, input: TensorView, scale: TensorView, bias: TensorView, n: number, h: number, c: number,
     epsilon: number) => {
      const components = getMaxComponents(c);
      const WG = 64;
      // we will store channel scale and channel shift in [2, components] matrix
      // or in vec2 when components == 1
      const outputType = components === 1 ? 'vec2f' : `mat2x${components}f`;
      const sumCastType = components === 1 ? 'f32' : `vec${components}f`;
      const setOutputValue = (var1: string, var2: string) => `${outputType}(${var1}, ${var2})`;
      const unitsOfWork = n * c / components;
      const wgSize = Math.ceil(h / WG);

      const meanInputDependencies: ProgramInputTensorInfoDependency[] = ['rank'];
      const meanProgramUniforms: ProgramUniform[] = [
        {type: 'uint32', data: wgSize}, {type: 'uint32', data: h}, {type: 'uint32', data: Math.floor(c / components)},
        {type: 'uint32', data: Math.floor(h * c / components)}
      ];

      const getMeanShaderSource = (shaderHelper: ShaderHelper) => {
        const inputHelper = inputVariable('input', input.dataType, input.dims, components);
        return `
  ${shaderHelper.declareVariables(inputHelper)}
  @group(0) @binding(1) var<storage, read_write> output : array<${outputType}>;
  struct Uniforms {wg_size:u32, H:u32, C:u32, image_size:u32};
  @group(0) @binding(2) var<uniform> uniforms: Uniforms;

  ${shaderHelper.mainStart(WG)}
    let currentImageNumber = global_idx / ${WG} / uniforms.C;
    let currentChannelNumber = (global_idx / ${WG}) % uniforms.C;
    let wgId = global_idx % ${WG};
    let wgOffset = wgId * uniforms.wg_size;
    if (wgOffset >= uniforms.H) {
        return;
    }
    let wgMax = min(wgOffset + uniforms.wg_size, uniforms.H);

    let offset = currentImageNumber * uniforms.image_size + currentChannelNumber;
    var sum = ${fillVector('f32', components)};
    var squaredSum = ${fillVector('f32', components)};
    for (var i: u32 = wgOffset; i < wgMax; i++) {
        let value = ${sumCastType}(input[offset + i * uniforms.C]);
        sum += value;
        squaredSum += value * value;
    }
    output[global_idx] = ${setOutputValue('sum', 'squaredSum')};
  }`;
      };

      const meanValues = context.compute(
          {
            name: 'InstanceNormComputeMean',
            shaderCache: {hint: `${components}`, inputDependencies: meanInputDependencies},
            getRunData: () => ({
              outputs: [
                {dims: [n, c, WG, 2], dataType: DataType.float},
              ],
              dispatchGroup: {x: n * c / components},
              programUniforms: meanProgramUniforms
            }),
            getShaderSource: getMeanShaderSource,
          },
          {inputs: [input], outputs: [-1]})[0];

      const programUniforms: ProgramUniform[] = [
        {type: 'uint32', data: unitsOfWork}, {type: 'uint32', data: h},
        {type: 'uint32', data: Math.floor(c / components)}, {type: 'uint32', data: Math.floor(WG * c / components)}
      ];
      const inputDependencies: ProgramInputTensorInfoDependency[] = ['rank', 'rank', 'rank'];
      const getShaderSource = (shaderHelper: ShaderHelper) => {
        const scaleHelper = inputVariable('scale', scale.dataType, scale.dims, components);
        const biasHelper = inputVariable('bias', bias.dataType, bias.dims, components);
        return `
  @group(0) @binding(0) var<storage, read> input : array<${outputType}>;
  @group(0) @binding(1) var<storage, read> scale : array<${scaleHelper.type.storage}>;
  @group(0) @binding(2) var<storage, read> bias : array<${biasHelper.type.storage}>;
  @group(0) @binding(3) var<storage, read_write> output : array<${outputType}>;
  struct Uniforms {units_of_work : u32, H: u32, C : u32, image_size : u32};
  @group(0) @binding(4) var<uniform> uniforms: Uniforms;

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.units_of_work')}
    let currentImageNumber = global_idx / uniforms.C;
    let currentChannelNumber = global_idx % uniforms.C;

    let offset = currentImageNumber * uniforms.image_size;
    var sum = ${fillVector('f32', components)};
    var squaredSum = ${fillVector('f32', components)};
    for (var i: u32 = 0; i < ${WG}; i++) {
        let value = input[offset + i + currentChannelNumber * ${WG}];
        sum += value[0];
        squaredSum += value[1];
    }
    sum = sum / f32(uniforms.H);
    squaredSum = squaredSum / f32(uniforms.H);
    let invStdDev = 1 / sqrt(squaredSum - sum * sum + f32(${epsilon}));
    let channelScale = invStdDev * ${sumCastType}(scale[currentChannelNumber]);
    let channelShift = ${sumCastType}(bias[currentChannelNumber]) - sum * channelScale;

    output[global_idx] = ${setOutputValue('channelScale', 'channelShift')};
  }`;
      };
      return context.compute(
          {
            name: 'InstanceNormComputeChannelScaleShift',
            // TODO: use epsilon as uniform. Currently epsilon as uniform fails test_instancenorm_epsilon.
            shaderCache: {hint: `${components};${epsilon}`, inputDependencies},
            getRunData: () => ({
              outputs: [
                {dims: [n, c, 2], dataType: DataType.float},
              ],
              dispatchGroup: {x: Math.ceil(unitsOfWork / 64 /* workgroup size */)},
              programUniforms
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
      const programUniforms: ProgramUniform[] =
          [{type: 'uint32', data: H}, {type: 'uint32', data: Math.floor(C / components)}];
      const inputDependencies: ProgramInputTensorInfoDependency[] = ['rank', 'rank'];

      // first compute mean
      const channelScaleShift = computeMean(context, inputs[0], inputs[1], inputs[2], N, H, C, attributes.epsilon);
      const getShaderSource = (shaderHelper: ShaderHelper) => {
        const dataType = tensorTypeToWsglStorageType(inputs[0].dataType);
        const scaleType = components === 1 ? 'vec2f' : `mat2x${components}f`;
        const scaleCastType = components === 1 ? dataType : `vec${components}<${dataType}>`;

        const inputHelper = inputVariable('input', inputs[0].dataType, inputs[0].dims, components);
        const outputHelper = outputVariable('output', inputs[0].dataType, outputShape, components);

        return `
  @group(0) @binding(0) var<storage, read> input : array<${inputHelper.type.storage}>;
  @group(0) @binding(1) var<storage, read> scaleInput : array<${scaleType}>;
  @group(0) @binding(2) var<storage, read_write> output : array<${outputHelper.type.storage}>;
  struct Uniforms {H: u32, C : u32};
  @group(0) @binding(3) var<uniform> uniforms: Uniforms;

  ${shaderHelper.mainStart()}
    let currentImageNumber = global_idx / (uniforms.C * uniforms.H);
    let currentChannelNumber = global_idx % uniforms.C;

    let scaleOffset = currentImageNumber * uniforms.C + currentChannelNumber;
    let scale = scaleInput[scaleOffset];
    output[global_idx] = fma(input[global_idx], ${scaleCastType}(scale[0]), ${scaleCastType}(scale[1]));
  }`;
      };
      context.compute(
          {
            name: 'InstanceNormalization',
            shaderCache: {hint: `${components}`, inputDependencies},
            getRunData: () => ({
              outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
              dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)},
              programUniforms
            }),
            getShaderSource,
          },
          {inputs: [inputs[0], channelScaleShift]});
    };

export const instanceNorm = (context: ComputeContext, attributes: InstanceNormAttributes): void => {
  if (attributes.format === 'NHWC') {
    createInstanceNormNHWCProgramInfo(context, context.inputs, attributes);
  } else {
    context.compute(createInstanceNormProgramInfo(context.inputs, attributes));
  }
};
