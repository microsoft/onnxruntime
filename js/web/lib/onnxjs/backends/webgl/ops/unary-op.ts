// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../../../attribute-with-cache-key';
import {Graph} from '../../../graph';
import {Tensor} from '../../../tensor';
import {FunctionType, GlslValueFunction} from '../glsl-definitions';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, ProgramInfoLoader, ProgramMetadata, TextureType} from '../types';

export function glslAbs(): GlslValueFunction {
  return glslBuiltinUnary('abs');
}
export function glslAcos(): GlslValueFunction {
  return glslBuiltinUnary('acos');
}
export function glslAsin(): GlslValueFunction {
  return glslBuiltinUnary('asin');
}
export function glslAtan(): GlslValueFunction {
  return glslBuiltinUnary('atan');
}
export function glslCeil(): GlslValueFunction {
  return glslBuiltinUnary('ceil');
}
export function glslCos(): GlslValueFunction {
  return glslBuiltinUnary('cos');
}
export function glslElu(alpha: number): GlslValueFunction {
  const name = 'elu';
  const body = `
  const float alpha = float(${alpha});

  float ${name}_(float a) {
    return a >= 0.0 ? a: (exp(a) - 1.0) * alpha;
  }
  vec4 ${name}_(vec4 v) {
    return vec4(${name}_(v.x), ${name}_(v.y), ${name}_(v.z), ${name}_(v.w));
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslExp(): GlslValueFunction {
  return glslBuiltinUnary('exp');
}
export function glslFloor(): GlslValueFunction {
  return glslBuiltinUnary('floor');
}
export function glslClip(min: number, max: number): GlslValueFunction {
  const name = 'clip';
  const body = `
  const float min = float(${min});
  const float max = float(${max});

  float ${name}_(float a) {
    return clamp(a, min, max);
  }
  vec4 ${name}_(vec4 v) {
    return clamp(v, min, max);
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslIdentity(): GlslValueFunction {
  const name = 'indentity';
  const body = `
  float ${name}_(float a) {
    return a;
  }
  vec4 ${name}_(vec4 v) {
    return v;
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslLeakyRelu(alpha: number): GlslValueFunction {
  const name = 'leakyRelu';
  const body = `
  const float alpha = float(${alpha});

  float ${name}_(float a) {
    return a < 0.0 ? a * alpha : a;
  }
  vec4 ${name}_(vec4 v) {
    return vec4(${name}_(v.x), ${name}_(v.y), ${name}_(v.z), ${name}_(v.w));
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslLog(): GlslValueFunction {
  return glslBuiltinUnary('log');
}
export function glslNeg(): GlslValueFunction {
  const name = 'neg';
  const body = `
  float ${name}_(float a) {
    return -a;
  }
  vec4 ${name}_(vec4 v) {
    return -v;
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslNot(): GlslValueFunction {
  const name = 'not';
  const body = `
  float ${name}_(float a) {
    return float( ! bool(a) );
  }
  bool ${name}_(bool a) {
    return !a;
  }
  vec4 ${name}_(vec4 v) {
    return vec4(!bool(v.x), !bool(v.y), !bool(v.z), !bool(v.w));
  }
  bvec4 ${name}_(bvec4 v) {
    return bvec4(!v.x, !v.y, !v.z, !v.w);
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslSin(): GlslValueFunction {
  return glslBuiltinUnary('sin');
}
export function glslRelu(): GlslValueFunction {
  const name = 'relu';
  const body = `
  float ${name}_(float a) {
    return max( a, 0.0 );
  }
  vec4 ${name}_(vec4 v) {
    return max( v, 0.0 );
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslSigmoid(): GlslValueFunction {
  const name = 'sigmoid';
  const body = `
  float ${name}_(float a) {
    return 1.0 / (1.0 + exp(-a));
  }
  vec4 ${name}_(vec4 v) {
    return 1.0 / (1.0 + exp(-v));
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslSqrt(): GlslValueFunction {
  return glslBuiltinUnary('sqrt');
}
export function glslTan(): GlslValueFunction {
  return glslBuiltinUnary('tan');
}
export function glslTanh(): GlslValueFunction {
  const name = 'tanh';
  const body = `
  float ${name}_(float a) {
    a = clamp(a, -10., 10.);
    a = exp(2.*a);
    return (a - 1.) / (a + 1.);
  }
  vec4 ${name}_(vec4 v) {
    v = clamp(v, -10., 10.);
    v = exp(2.*v);
    return (v - 1.) / (v + 1.);
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
function glslBuiltinUnary(name: string): GlslValueFunction {
  const body = `
  float ${name}_(float a) {
    return ${name}(a);
  }
  vec4 ${name}_(vec4 v) {
    return ${name}(v);
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}

/////
/////
/////

const createElementwiseProgramInfo =
    (handler: WebGLInferenceHandler, metadata: ProgramMetadata, input: Tensor, glslFunc: GlslValueFunction):
        ProgramInfo => {
          const textureType = handler.session.pack ? TextureType.packed : TextureType.unpacked;
          const glsl = getGlsl(handler.session.backend.glContext.version);
          return {
            ...metadata,
            output: {dims: input.dims, type: input.type, textureType},
            shaderSource: `
     ${glslFunc.body}
     void main() {
       vec4 v = ${glsl.texture2D}(A, TexCoords);
       v = ${glslFunc.name}_(v);
       ${glsl.output} = v;
     }
     `,
            hasMain: true
          };
        };

const createElementwiseProgramInfoLoader =
    (handler: WebGLInferenceHandler, input: Tensor, glslFunc: GlslValueFunction, cacheKey?: string):
        ProgramInfoLoader => {
          const textureType = handler.session.pack ? TextureType.packed : TextureType.unpacked;
          const metadata = {name: glslFunc.name, inputTypes: [textureType], inputNames: ['A'], cacheHint: cacheKey};
          return {...metadata, get: () => createElementwiseProgramInfo(handler, metadata, input, glslFunc)};
        };

export const abs = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslAbs()), inputs)];

export const acos = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslAcos()), inputs)];

export const asin = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslAsin()), inputs)];

export const atan = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslAtan()), inputs)];

export interface ClipAttributes extends AttributeWithCacheKey {
  readonly min: number;
  readonly max: number;
}

export const clip =
    (handler: WebGLInferenceHandler, inputs: Tensor[], attributes: ClipAttributes): Tensor[] => [handler.run(
        createElementwiseProgramInfoLoader(
            handler, inputs[0], glslClip(attributes.min, attributes.max), attributes.cacheKey),
        inputs)];

export const parseClipAttributes = (node: Graph.Node): ClipAttributes => createAttributeWithCacheKey({
  min: node.attributes.getFloat('min', -3.4028234663852886e+38),
  max: node.attributes.getFloat('max', 3.4028234663852886e+38)
});

export const ceil = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslCeil()), inputs)];

export const cos = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslCos()), inputs)];

export interface EluAttributes extends AttributeWithCacheKey {
  readonly alpha: number;
}

export const elu =
    (handler: WebGLInferenceHandler, inputs: Tensor[], attributes: EluAttributes): Tensor[] => [handler.run(
        createElementwiseProgramInfoLoader(handler, inputs[0], glslElu(attributes.alpha), attributes.cacheKey),
        inputs)];

export const parseEluAttributes = (node: Graph.Node): EluAttributes =>
    createAttributeWithCacheKey({alpha: node.attributes.getFloat('alpha', 1.0)});

export const exp = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslExp()), inputs)];

export const floor = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslFloor()), inputs)];

export const identity = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslIdentity()), inputs)];

export interface LeakyReluAttributes extends AttributeWithCacheKey {
  readonly alpha: number;
}

export const leakyRelu =
    (handler: WebGLInferenceHandler, inputs: Tensor[], attributes: LeakyReluAttributes): Tensor[] => [handler.run(
        createElementwiseProgramInfoLoader(handler, inputs[0], glslLeakyRelu(attributes.alpha), attributes.cacheKey),
        inputs)];

export const parseLeakyReluAttributes = (node: Graph.Node): LeakyReluAttributes =>
    createAttributeWithCacheKey({alpha: node.attributes.getFloat('alpha', 0.01)});

export const log = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslLog()), inputs)];

export const neg = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslNeg()), inputs)];

export const not = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslNot()), inputs)];

export const relu = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslRelu()), inputs)];

export const sigmoid = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslSigmoid()), inputs)];

export const sin = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslSin()), inputs)];

export const sqrt = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslSqrt()), inputs)];

export const tan = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslTan()), inputs)];

export const tanh = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createElementwiseProgramInfoLoader(handler, inputs[0], glslTanh()), inputs)];
