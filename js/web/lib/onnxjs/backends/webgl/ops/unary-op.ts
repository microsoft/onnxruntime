// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {UnaryOp} from '../../../ops/unary-op';
import {Tensor} from '../../../tensor';
import {FunctionType, GlslValueFunction} from '../glsl-definitions';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

export class WebGLUnaryOp extends UnaryOp implements WebGLOperator {
  constructor(protected typeConstraint: readonly Tensor.DataType[], protected glslFunc: GlslValueFunction) {
    super(typeConstraint);
  }
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const outputShape = inputs[0].dims.slice();
    const inputLayout = handler.getOrCreateTextureLayout(inputs[0]);
    const glsl = getGlsl(handler.session.backend.glContext.version);
    const shaderSource = `
      ${this.glslFunc.body}
      void main() {
        vec4 v = ${glsl.texture2D}(A, TexCoords);
        v = ${this.glslFunc.name}(v);
        ${glsl.output} = v;
      }
      `;
    const outputLayout = handler.createTextureLayoutFromShape(outputShape);
    return {inputLayouts: [inputLayout], outputLayout, samplers: ['A'], shaderSource, hasMain: true};
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = [handler.getOrCreateTextureData(inputs[0], programInfo.inputLayouts[0])];
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}

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
export function glslExp(): GlslValueFunction {
  return glslBuiltinUnary('exp');
}
export function glslFloor(): GlslValueFunction {
  return glslBuiltinUnary('floor');
}
export function glslIdentity(): GlslValueFunction {
  const name = 'indentity_';
  const body = `
  float ${name}(float a) {
    return a;
  }
  vec4 ${name}(vec4 v) {
    return v;
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslLog(): GlslValueFunction {
  return glslBuiltinUnary('log');
}
export function glslNeg(): GlslValueFunction {
  const name = 'neg_';
  const body = `
  float ${name}(float a) {
    return -a;
  }
  vec4 ${name}(vec4 v) {
    return -v;
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslNot(): GlslValueFunction {
  const name = 'not_';
  const body = `
  float ${name}(float a) {
    return float( ! bool(a) );
  }
  bool ${name}(bool a) {
    return !a;
  }
  vec4 ${name}(vec4 v) {
    return vec4(!bool(v.x), !bool(v.y), !bool(v.z), !bool(v.w));
  }
  bvec4 ${name}(bvec4 v) {
    return bvec4(!v.x, !v.y, !v.z, !v.w);
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslSin(): GlslValueFunction {
  return glslBuiltinUnary('sin');
}
export function glslRelu(): GlslValueFunction {
  const name = 'relu_';
  const body = `
  float ${name}(float a) {
    return max( a, 0.0 );
  }
  vec4 ${name}(vec4 v) {
    return max( v, 0.0 );
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslSigmoid(): GlslValueFunction {
  const name = 'sigmoid_';
  const body = `
  float ${name}(float a) {
    return 1.0 / (1.0 + exp(-a));
  }
  vec4 ${name}(vec4 v) {
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
  const name = 'tanh_';
  const body = `
  float ${name}(float a) {
    a = clamp(a, -10., 10.);
    a = exp(2.*a);
    return (a - 1.) / (a + 1.);
  }
  vec4 ${name}(vec4 v) {
    v = clamp(v, -10., 10.);
    v = exp(2.*v);
    return (v - 1.) / (v + 1.);
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
function glslBuiltinUnary(fname: string): GlslValueFunction {
  const name = `${fname}_`;
  const body = `
  float ${name}(float a) {
    return ${fname}(a);
  }
  vec4 ${name}(vec4 v) {
    return ${fname}(v);
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
