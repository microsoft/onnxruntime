// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {WORKGROUP_SIZE} from './common';

type BuiltinFunctionName = string;
type ElementwiseCustomExpression = (expression: string) => string;
type ElementwiseFunctionCall = BuiltinFunctionName|ElementwiseCustomExpression;

const createElementwiseProgramShader =
    (datasize: number, funcCall: ElementwiseFunctionCall, additionalImplementation?: string): string => {
      const vecSize = Math.ceil(datasize / 4);

      let expression = '';
      if (typeof funcCall === 'string') {
        expression = `${funcCall}(a)`;
      } else {
        expression = funcCall('a');
      }
      return `
  const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;

  @group(0) @binding(0) var<storage, read> inputData : array<vec4<f32>>;
  @group(0) @binding(1) var<storage, read_write> outputData : array<vec4<f32>>;

  ${additionalImplementation ?? ''}

  @compute @workgroup_size(WORKGROUP_SIZE)
  fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {

    // Guard against out-of-bounds work group sizes
    if (global_id.x >= ${vecSize}u) {
      return;
    }

    let a = inputData[global_id.x];
    outputData[global_id.x] = ${expression};
  }`;
    };

const createElementwiseProgramInfo =
    (metadata: ProgramMetadata, input: TensorView, funcCall: ElementwiseFunctionCall,
     additionalImplementation?: string): ProgramInfo => ({
      ...metadata,
      shaderSource: createElementwiseProgramShader(ShapeUtil.size(input.dims), funcCall, additionalImplementation),
      outputs: [{dims: input.dims, dataType: input.dataType, gpuDataType: GpuDataType.default}],
      dispatchGroup: (inputTensors) =>
          ({x: Math.ceil(ShapeUtil.size(inputTensors[0].dims) / 64 /* workgroup size */ / 4 /* vec size */)})
    });

const createElementwiseProgramInfoLoader =
    (input: TensorView, name: string, funcCall: ElementwiseFunctionCall, additionalImplementation?: string,
     cacheKey?: string): ProgramInfoLoader => {
      const metadata: ProgramMetadata = {name, inputTypes: [GpuDataType.default], cacheHint: cacheKey};
      return {
        ...metadata,
        get: () => createElementwiseProgramInfo(metadata, input, funcCall, additionalImplementation)
      };
    };

export const abs = (context: ComputeContext): number =>
    context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Abs', 'abs'));

export const acos = (context: ComputeContext): number =>
    context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Acos', 'acos'));

export const asin = (context: ComputeContext): number =>
    context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Asin', 'asin'));

export const atan = (context: ComputeContext): number =>
    context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Atan', 'atan'));

export interface ClipAttributes extends AttributeWithCacheKey {
  readonly min: number;
  readonly max: number;
}

export const clip = (context: ComputeContext, attributes: ClipAttributes): number =>
    context.compute(createElementwiseProgramInfoLoader(
        context.inputs[0], 'Clip', a => `clamp(${a}, clip_min_, clip_max_)`, `
    let clip_min_: vec4<f32> = vec4(f32(${attributes.min}));
    let clip_max_: vec4<f32> = vec4(f32(${attributes.max}));
`,
        attributes.cacheKey));

// export const parseClipAttributes = (node: Graph.Node): ClipAttributes => createAttributeWithCacheKey(
//     {min: node.attributes.getFloat('min', MIN_CLIP), max: node.attributes.getFloat('max', MAX_CLIP)});

// const generateClipAttributesFromInputs = (handler: WebGpuInferenceHandler, inputs: Tensor[]): ClipAttributes => {
//   if (inputs.length >= 3 &&
//       (!handler.session.isInitializer(inputs[1].dataId) || !handler.session.isInitializer(inputs[2].dataId))) {
//     throw new Error('dynamic clip attributes are not allowed');
//   }

//   const min = (inputs.length >= 3) ? inputs[1].numberData[0] : MIN_CLIP;
//   const max = (inputs.length >= 3) ? inputs[2].numberData[0] : MAX_CLIP;
//   return createAttributeWithCacheKey({min, max});
// };

// export const clipV11 = (context: ComputeContext ): number=> {
//   const attributes = generateClipAttributesFromInputs(handler, inputs);
//   return clip(handler, [inputs[0]], attributes);
// };

// export const ceil = async(handler: WebGpuInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> =>
//     handler.run(createElementwiseProgramInfoLoader(inputs[0], 'Ceil', 'ceil'), inputs);

// export const cos = async(handler: WebGpuInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> =>
//     handler.run(createElementwiseProgramInfoLoader(inputs[0], 'Cos', 'cos'), inputs);

// export interface EluAttributes extends AttributeWithCacheKey {
//   readonly alpha: number;
// }

// export const elu = async(handler: WebGpuInferenceHandler, inputs: Tensor[], attributes: EluAttributes):
//                        Promise<Tensor[] >=>handler.run(
//                            createElementwiseProgramInfoLoader(
//                                inputs[0], 'Elu', a => `elu_vf32(${a})`, `
//     let elu_alpha_: f32 = f32(${attributes.alpha});

//     fn elu_f32(a: f32) -> f32 {
//       return select((exp(a) - 1.0) * elu_alpha_, a, a >= 0.0);
//     }

//     fn elu_vf32(v: vec4<f32>) -> vec4<f32> {
//       return vec4(elu_f32(v.x), elu_f32(v.y), elu_f32(v.z), elu_f32(v.w));
//     }`,
//                                attributes.cacheKey),
//                            inputs);

// export const parseEluAttributes = (node: Graph.Node): EluAttributes =>
//     createAttributeWithCacheKey({alpha: node.attributes.getFloat('alpha', 1.0)});

// export const exp = async(handler: WebGpuInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> =>
//     handler.run(createElementwiseProgramInfoLoader(inputs[0], 'Exp', 'exp'), inputs);

// export const floor = async(handler: WebGpuInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> =>
//     handler.run(createElementwiseProgramInfoLoader(inputs[0], 'Floor', 'floor'), inputs);

// export interface LeakyReluAttributes extends AttributeWithCacheKey {
//   readonly alpha: number;
// }

// export const leakyRelu = async(handler: WebGpuInferenceHandler, inputs: Tensor[], attributes: EluAttributes):
//                              Promise<Tensor[] >=>handler.run(
//                                  createElementwiseProgramInfoLoader(
//                                      inputs[0], 'LeakyRelu', a => `leaky_relu_vf32(${a})`, `
//     let leaky_relu_alpha_: f32 = f32(${attributes.alpha});

//     fn leaky_relu_f32(a: f32) -> f32 {
//       return select(a, a * leaky_relu_alpha_, a < 0.0);
//     }

//     fn leaky_relu_vf32(v: vec4<f32>) -> vec4<f32> {
//       return vec4(leaky_relu_f32(v.x), leaky_relu_f32(v.y), leaky_relu_f32(v.z), leaky_relu_f32(v.w));
//     }`,
//                                      attributes.cacheKey),
//                                  inputs);

// export const parseLeakyReluAttributes = (node: Graph.Node): LeakyReluAttributes =>
//     createAttributeWithCacheKey({alpha: node.attributes.getFloat('alpha', 0.01)});

// export const log = async(handler: WebGpuInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> =>
//     handler.run(createElementwiseProgramInfoLoader(inputs[0], 'Log', 'log'), inputs);

// export const neg = async(handler: WebGpuInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> =>
//     handler.run(createElementwiseProgramInfoLoader(inputs[0], 'Neg', a => `-${a}`), inputs);

// // export const not = (handler: WebGLInferenceHandler, inputs: Tensor[]):
// //     Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslNot()), inputs)];

// export const relu = async(handler: WebGpuInferenceHandler, inputs: Tensor[]): Promise<Tensor[] >=>handler.run(
//     createElementwiseProgramInfoLoader(inputs[0], 'Relu', a => `max(${a}, vec4(0.0))`), inputs);

// export const sigmoid = async(handler: WebGpuInferenceHandler, inputs: Tensor[]): Promise<Tensor[] >=>handler.run(
//     createElementwiseProgramInfoLoader(inputs[0], 'Sigmoid', a => `(vec4(1.0) / (vec4(1.0) + exp(-${a})))`), inputs);

// export const sin = async(handler: WebGpuInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> =>
//     handler.run(createElementwiseProgramInfoLoader(inputs[0], 'Sin', 'sin'), inputs);

// export const sqrt = async(handler: WebGpuInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> =>
//     handler.run(createElementwiseProgramInfoLoader(inputs[0], 'Sqrt', 'sqrt'), inputs);

// export const tan = async(handler: WebGpuInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> =>
//     handler.run(createElementwiseProgramInfoLoader(inputs[0], 'Tan', 'tan'), inputs);

// export const tanh = async(handler: WebGpuInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> =>
//     handler.run(createElementwiseProgramInfoLoader(inputs[0], 'Tanh', 'tanh'), inputs);
