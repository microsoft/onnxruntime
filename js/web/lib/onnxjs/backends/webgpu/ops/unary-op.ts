// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../../../attribute-with-cache-key';
import {Graph} from '../../../graph';
import {Tensor} from '../../../tensor';
import {MAX_CLIP, MIN_CLIP} from '../../../util';
import {WebGpuInferenceHandler} from '../inference-handler';
import {GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

const createElementwiseProgramShader = (funcName: string, funcImpl: string): (datasize: number) => string =>
    (datasize) => {
      const vecSize = Math.ceil(datasize / 4);
      return `
  // constant value for a workgroup size.
  //
  // We definitely can do further optimization in future, but for now we use 64.
  //
  // rule of thumb: Use [a workgroup size of] 64 unless you know what GPU you are targeting or that your workload
  //                needs something different.
  //
  // from: https://surma.dev/things/webgpu/
  //
  let WORKGROUP_SIZE: u32 = 64u;

  @group(0) @binding(0) var<storage, read> inputData : array<vec4<f32>>;
  @group(0) @binding(1) var<storage, write> outputData : array<vec4<f32>>;

  ${funcImpl}

  @stage(compute) @workgroup_size(WORKGROUP_SIZE)
  fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {

    // Guard against out-of-bounds work group sizes
    if (global_id.x >= ${vecSize}u) {
      return;
    }

    outputData[global_id.x] = ${funcName}(inputData[global_id.x]);
  }`;
    };

const createElementwiseProgramInfo =
    (metadata: ProgramMetadata, input: Tensor, funcName: string, funcImpl = ''): ProgramInfo => ({
      ...metadata,
      shaderSource: createElementwiseProgramShader(funcName, funcImpl)(input.size),
      outputs: [{dims: input.dims, type: input.type, gpuDataType: GpuDataType.default}],
      dispatchGroup: (inputTensors) =>
          ({x: Math.ceil(inputTensors[0].size / 64 /* workgroup size */ / 4 /* vec size */)})
    });

const createElementwiseProgramInfoLoader =
    (input: Tensor, functionName: string, functionImplementation = '', cacheKey?: string): ProgramInfoLoader => {
      const metadata: ProgramMetadata = {name: functionName, inputTypes: [GpuDataType.default], cacheHint: cacheKey};
      return {
        ...metadata,
        get: () => createElementwiseProgramInfo(metadata, input, functionName, functionImplementation)
      };
    };

export const abs = (handler: WebGpuInferenceHandler, inputs: Tensor[]): Tensor[] =>
    handler.run(createElementwiseProgramInfoLoader(inputs[0], 'abs'), inputs);

export const acos = (handler: WebGpuInferenceHandler, inputs: Tensor[]): Tensor[] =>
    handler.run(createElementwiseProgramInfoLoader(inputs[0], 'acos'), inputs);

export const asin = (handler: WebGpuInferenceHandler, inputs: Tensor[]): Tensor[] =>
    handler.run(createElementwiseProgramInfoLoader(inputs[0], 'asin'), inputs);

export const atan = (handler: WebGpuInferenceHandler, inputs: Tensor[]): Tensor[] =>
    handler.run(createElementwiseProgramInfoLoader(inputs[0], 'atan'), inputs);

export interface ClipAttributes extends AttributeWithCacheKey {
  readonly min: number;
  readonly max: number;
}

export const clip = (handler: WebGpuInferenceHandler, inputs: Tensor[], attributes: ClipAttributes): Tensor[] =>
    handler.run(
        createElementwiseProgramInfoLoader(
            inputs[0], 'clip', `
    let clip_min_: vec4<f32> = vec4(f32(${attributes.min}));
    let clip_max_: vec4<f32> = vec4(f32(${attributes.max}));

    fn clip(x: vec4<f32>) -> vec4<f32> {
      return clamp(x, clip_min_, clip_max_);
    }`,
            attributes.cacheKey),
        inputs);

export const parseClipAttributes = (node: Graph.Node): ClipAttributes => createAttributeWithCacheKey(
    {min: node.attributes.getFloat('min', MIN_CLIP), max: node.attributes.getFloat('max', MAX_CLIP)});

const generateClipAttributesFromInputs = (handler: WebGpuInferenceHandler, inputs: Tensor[]): ClipAttributes => {
  if (inputs.length >= 3 &&
      (!handler.session.isInitializer(inputs[1].dataId) || !handler.session.isInitializer(inputs[2].dataId))) {
    throw new Error('dynamic clip attributes are not allowed');
  }

  const min = (inputs.length >= 3) ? inputs[1].numberData[0] : MIN_CLIP;
  const max = (inputs.length >= 3) ? inputs[2].numberData[0] : MAX_CLIP;
  return createAttributeWithCacheKey({min, max});
};

export const clipV11 = (handler: WebGpuInferenceHandler, inputs: Tensor[]): Tensor[] => {
  const attributes = generateClipAttributesFromInputs(handler, inputs);
  return clip(handler, [inputs[0]], attributes);
};

// export const ceil = (handler: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslCeil()), inputs)];

// export const cos = (handler: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslCos()), inputs)];

// export interface EluAttributes extends AttributeWithCacheKey {
//   readonly alpha: number;
// }

// export const elu =
//     (handler: WebGLInferenceHandler, inputs: Tensor[], attributes: EluAttributes): Tensor[] => [handler.run(
//         createElementwiseProgramInfoLoader(handler, inputs[0], glslElu(attributes.alpha), attributes.cacheKey),
//         inputs)];

// export const parseEluAttributes = (node: Graph.Node): EluAttributes =>
//     createAttributeWithCacheKey({alpha: node.attributes.getFloat('alpha', 1.0)});

// export const exp = (handler: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslExp()), inputs)];

// export const floor = (handler: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslFloor()), inputs)];

// export const identity = (handler: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslIdentity()), inputs)];

// export interface LeakyReluAttributes extends AttributeWithCacheKey {
//   readonly alpha: number;
// }

// export const leakyRelu =
//     (handler: WebGLInferenceHandler, inputs: Tensor[], attributes: LeakyReluAttributes): Tensor[] => [handler.run(
//         createElementwiseProgramInfoLoader(handler, inputs[0], glslLeakyRelu(attributes.alpha), attributes.cacheKey),
//         inputs)];

// export const parseLeakyReluAttributes = (node: Graph.Node): LeakyReluAttributes =>
//     createAttributeWithCacheKey({alpha: node.attributes.getFloat('alpha', 0.01)});

// export const log = (handler: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslLog()), inputs)];

// export const neg = (handler: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslNeg()), inputs)];

// export const not = (handler: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslNot()), inputs)];

// export const relu = (handler: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslRelu()), inputs)];

// export const sigmoid = (handler: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslSigmoid()), inputs)];

// export const sin = (handler: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslSin()), inputs)];

// export const sqrt = (handler: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslSqrt()), inputs)];

// export const tan = (handler: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslTan()), inputs)];

// export const tanh = (handler: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslTanh()), inputs)];
