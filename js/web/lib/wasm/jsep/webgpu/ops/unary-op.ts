// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {MAX_CLIP, MIN_CLIP, ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {inputVariable, outputVariable, ShaderHelper} from './common';

type BuiltinFunctionName = string;
type ElementwiseCustomExpression = (expression: string) => string;
type ElementwiseFunctionCall = BuiltinFunctionName|ElementwiseCustomExpression;

const createElementwiseProgramShader =
    (shaderHelper: ShaderHelper, datasize: number, inputDataType: number, outputDataType: number,
     funcCall: ElementwiseFunctionCall, additionalImplementation?: string): string => {
      const vecSize = Math.ceil(datasize / 4);

      let expression = '';
      if (typeof funcCall === 'string') {
        expression = `${funcCall}(a)`;
      } else {
        expression = funcCall('a');
      }

      const input = inputVariable('inputData', inputDataType, [vecSize], 4);
      const output = outputVariable('outputData', outputDataType, [vecSize], 4);

      return `
  ${shaderHelper.declareVariables(input, output)}

  ${additionalImplementation ?? ''}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(vecSize)}

    let a = ${input.getByOffset('global_idx')};
    ${output.setByOffset('global_idx', expression)}
  }`;
    };

const createElementwiseProgramInfo =
    (metadata: ProgramMetadata, input: TensorView, outputDataType: number, funcCall: ElementwiseFunctionCall,
     additionalImplementation?: string): ProgramInfo => ({
      ...metadata,
      getShaderSource: shaderHelper => createElementwiseProgramShader(
          shaderHelper, ShapeUtil.size(input.dims), input.dataType, outputDataType, funcCall, additionalImplementation),
      outputs: [{dims: input.dims, dataType: outputDataType, gpuDataType: GpuDataType.default}],
      dispatchGroup: (inputTensors) =>
          ({x: Math.ceil(ShapeUtil.size(inputTensors[0].dims) / 64 /* workgroup size */ / 4 /* vec size */)})
    });

const createElementwiseProgramInfoLoader =
    (input: TensorView, name: string, funcCall: ElementwiseFunctionCall, additionalImplementation?: string,
     cacheKey?: string, outputDataType: number = input.dataType): ProgramInfoLoader => {
      const metadata: ProgramMetadata = {name, inputTypes: [GpuDataType.default], cacheHint: cacheKey};
      return {
        ...metadata,
        get: () => createElementwiseProgramInfo(metadata, input, outputDataType, funcCall, additionalImplementation)
      };
    };

export const abs = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Abs', 'abs'));
};

export const acos = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Acos', 'acos'));
};

export const acosh = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Acosh', 'acosh'));
};

export const asin = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Asin', 'asin'));
};

export const asinh = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Asinh', 'asinh'));
};

export const atan = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Atan', 'atan'));
};
export const atanh = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Atanh', 'atanh'));
};

export interface CastAttributes extends AttributeWithCacheKey {
  readonly to: number;
  readonly saturate?: boolean;
}

export const parseCastAttributes = (attributes: Record<string, unknown>): CastAttributes =>
    createAttributeWithCacheKey(attributes as {to: number});


export const cast = (context: ComputeContext, attributes: CastAttributes): void => {
  let func: ElementwiseFunctionCall;
  switch (attributes.to) {
    case DataType.float:
      func = 'vec4<f32>';
      break;
    case DataType.uint32:
      func = 'vec4<u32>';
      break;
    case DataType.int32:
      func = 'vec4<i32>';
      break;
    case DataType.bool:
      func = 'vec4<bool>';
      break;
    default:
      throw new RangeError(`not supported type (specified in attribute 'to' from 'Cast' operator): ${attributes.to}`);
  }
  context.compute(createElementwiseProgramInfoLoader(
      context.inputs[0], 'Cast', func, undefined, attributes.cacheKey, attributes.to));
};

export interface ClipAttributes extends AttributeWithCacheKey {
  readonly min: number;
  readonly max: number;
}

export const clipV10 = (context: ComputeContext, attributes: ClipAttributes): void => {
  context.compute(
      createElementwiseProgramInfoLoader(
          context.inputs[0], 'Clip', a => `clamp(${a}, clip_min_, clip_max_)`, `
    const clip_min_: vec4<f32> = vec4(f32(${attributes.min}));
    const clip_max_: vec4<f32> = vec4(f32(${attributes.max}));
`,
          attributes.cacheKey),
      {inputs: [0]});
};
const generateClipAttributesFromInputs = (inputs: readonly TensorView[]): ClipAttributes => {
  const min = (inputs.length >= 2) ? inputs[1].getFloat32Array()[0] : MIN_CLIP;
  const max = (inputs.length >= 3) ? inputs[2].getFloat32Array()[0] : MAX_CLIP;
  return createAttributeWithCacheKey({min, max});
};

export const clip = (context: ComputeContext): void => {
  const attributes = generateClipAttributesFromInputs(context.inputs);
  clipV10(context, attributes);
};

export const ceil = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Ceil', 'ceil'));
};

export const cos = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Cos', 'cos'));
};

export const cosh = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Cosh', 'cosh'));
};

export interface AlphaAttributes extends AttributeWithCacheKey {
  readonly alpha: number;
}

export const parseAlphaAttributes = (attributes: Record<string, unknown>): AlphaAttributes =>
    createAttributeWithCacheKey(attributes as {alpha: number});

export const elu = (context: ComputeContext, attributes: AlphaAttributes): void => {
  context.compute(createElementwiseProgramInfoLoader(
      context.inputs[0], 'Elu', a => `elu_vf32(${a})`, `
  const elu_alpha_: f32 = f32(${attributes.alpha});

  fn elu_f32(a: f32) -> f32 {
  return select((exp(a) - 1.0) * elu_alpha_, a, a >= 0.0);
  }

  fn elu_vf32(v: vec4<f32>) -> vec4<f32> {
  return vec4(elu_f32(v.x), elu_f32(v.y), elu_f32(v.z), elu_f32(v.w));
  }`,
      attributes.cacheKey));
};

export const erfImpl = (dataType: string) => `
const r0: f32 = 0.3275911;
const r1: f32 = 0.254829592;
const r2: f32 = -0.284496736;
const r3: f32 = 1.421413741;
const r4: f32 = -1.453152027;
const r5: f32 = 1.061405429;

fn erf_vf32(v: ${dataType}) -> ${dataType} {
  let absv = abs(v);
  let x = 1.0 / (1.0 + r0 * absv);
  return sign(v) * (1.0 - ((((r5 * x + r4) * x + r3) * x + r2) * x + r1) * x * exp(-absv * absv));
}`;

export const erf = (context: ComputeContext): void => {
  context.compute(
      createElementwiseProgramInfoLoader(context.inputs[0], 'Erf', a => `erf_vf32(${a})`, erfImpl('vec4<f32>')));
};

export const exp = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Exp', 'exp'));
};

export const floor = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Floor', 'floor'));
};

export const gelu = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(
      context.inputs[0], 'Gelu', a => `0.5 * ${a} * (1.0 + erf_vf32(${a} * 0.7071067811865475))`,
      erfImpl('vec4<f32>')));
};

export const leakyRelu = (context: ComputeContext, attributes: AlphaAttributes): void => {
  context.compute(createElementwiseProgramInfoLoader(
      context.inputs[0], 'LeakyRelu', a => `select(leaky_relu_alpha_ * ${a}, ${a}, ${a} >= vec4<f32>(0.0))`,
      `const leaky_relu_alpha_: f32 = f32(${attributes.alpha});`, attributes.cacheKey));
};

export const neg = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Neg', a => `-${a}`));
};

export const reciprocal = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Reciprocal', a => `1.0/${a}`));
};

export const relu = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(
      context.inputs[0], 'Relu', a => `select(vec4<f32>(0.0), ${a}, ${a} > vec4<f32>(0.0))`));
};

export const sigmoid = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Sigmoid', a => `(1.0 / (1.0 + exp(-${a})))`));
};

export const sin = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Sin', 'sin'));
};

export const sinh = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Sinh', 'sinh'));
};

export const sqrt = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Sqrt', 'sqrt'));
};

export const tan = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Tan', 'tan'));
};

export const tanh = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Tanh', 'tanh'));
};

export const thresholdedRelu = (context: ComputeContext, attributes: AlphaAttributes): number => {
  context.compute(createElementwiseProgramInfoLoader(
      context.inputs[0], 'ThresholdedRelu', a => `select(vec4<f32>(0.0), ${a}, ${a} > thresholded_relu_alpha_)`,
      `const thresholded_relu_alpha_: vec4<f32> = vec4<f32>(${attributes.alpha});`, attributes.cacheKey));
  return 0;
};

export const log = (context: ComputeContext): void => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Log', 'log'));
};
