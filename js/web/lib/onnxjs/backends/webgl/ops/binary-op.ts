// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {BinaryOp} from '../../../ops/binary-op';
import {Tensor} from '../../../tensor';
import {BroadcastUtil, ShapeUtil} from '../../../util';
import {FunctionType, GlslValueFunction} from '../glsl-definitions';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

export class WebGLBinaryOp extends BinaryOp implements WebGLOperator {
  private usePackedTexture?: boolean;

  constructor(
      typeConstraint: readonly Tensor.DataType[], protected glslFunc: GlslValueFunction, opType?: string,
      resultType?: Tensor.DataType) {
    super(typeConstraint, opType, resultType);
  }
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const isBroadcast = !ShapeUtil.areEqual(inputs[0].dims, inputs[1].dims);

    // TODO fix bcast in packed mode.
    if (this.usePackedTexture === undefined) {
      this.usePackedTexture = !isBroadcast && handler.session.pack;
    }

    const inputLayouts = this.usePackedTexture ?
        inputs.map(t => handler.getOrCreateTextureLayout(t, 4, true, t.dims, true)) :
        inputs.map(t => handler.getOrCreateTextureLayout(t));
    const ouputLayout = this.usePackedTexture ?
        handler.createTextureLayoutFromShape(inputs[0].dims, 4, inputs[0].dims, {isPacked: true, reverseWH: true}) :
        handler.createTextureLayoutFromShape(inputs[0].dims);

    if (isBroadcast) {
      const outputShape = BroadcastUtil.calcShape(inputs[0].dims, inputs[1].dims, false);
      if (!outputShape) {
        throw new Error('Can\'t perform binary op on the given tensors');
      }
      const outputRank = outputShape.length;
      const aRank = inputs[0].dims.length !== 0 ? inputs[0].dims.length : 1;
      const bRank = inputs[1].dims.length !== 0 ? inputs[1].dims.length : 1;
      const aBcast = inputs[0].dims.length !== 0 ? 'bcastIndices_A(indices, aindices);' : 'aindices[0] = 0;';
      const bBcast = inputs[1].dims.length !== 0 ? 'bcastIndices_B(indices, bindices);' : 'bindices[0] = 0;';

      // TODO: for packed tensors, we need to implement logic to caculate textCoords for broadcast tensor
      const shaderSource = `
      ${this.glslFunc.body}
      float process(int indices[${outputRank}]) {
        int aindices[${aRank}];
        int bindices[${bRank}];
        ${aBcast}
        ${bBcast}
        return ${this.glslFunc.name}(_A(aindices), _B(bindices));
    }`;
      const outputLayout = this.usePackedTexture ?
          handler.createTextureLayoutFromShape(outputShape, 4, outputShape, {isPacked: true, reverseWH: true}) :
          handler.createTextureLayoutFromShape(outputShape);

      return {
        inputLayouts,
        outputLayout,
        samplers: ['A', 'B'],
        shaderSource,
        expectPackedInputs: this.usePackedTexture,
        expectPackedOutputs: this.usePackedTexture
      };
    }
    const glsl = getGlsl(handler.session.backend.glContext.version);
    const shaderSource = `
    ${this.glslFunc.body}
    void main() {
      vec4 v1 = ${glsl.texture2D}(A, TexCoords);
      vec4 v2 = ${glsl.texture2D}(B, TexCoords);
      vec4 result = ${this.glslFunc.name}(v1, v2);
      ${glsl.output} = result;
    }
    `;

    if (this.usePackedTexture) {
      return {
        hasMain: true,
        inputLayouts,
        outputLayout: ouputLayout,
        samplers: ['A', 'B'],
        shaderSource,
        expectPackedInputs: true,
        expectPackedOutputs: true
      };
    } else {
      return {
        hasMain: true,
        inputLayouts,
        outputLayout: handler.createTextureLayoutFromShape(inputs[0].dims),
        samplers: ['A', 'B'],
        shaderSource,
      };
    }
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = this.usePackedTexture ?
        inputs.map((t) => handler.getOrCreateTextureData(t, handler.getOrCreateTextureLayout(t, 1, false, [], true))) :
        inputs.map((t, i) => handler.getOrCreateTextureData(t, programInfo.inputLayouts[i]));
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(
          programInfo.outputLayout, this.resultType ? this.resultType : inputs[0].type),
      uniformData: {}
    };
  }
}

export function glslAdd(): GlslValueFunction {
  const name = 'add_';
  const body = `
  float ${name}(float a, float b) {
    return a + b;
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    return v1 + v2;
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslDiv(): GlslValueFunction {
  const name = 'div_';
  const body = `
  float ${name}(float a, float b) {
    return a / b;
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    return v1 / v2;
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslMul(): GlslValueFunction {
  const name = 'mul_';
  const body = `
  float ${name}(float a, float b) {
    return a * b;
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    return v1 * v2;
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslSub(): GlslValueFunction {
  const name = 'sub_';
  const body = `
  float ${name}(float a, float b) {
    return a - b;
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    return v1 - v2;
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslEqual(): GlslValueFunction {
  const name = 'equal_';
  const body = `
  float ${name}(float a, float b) {
    return float(a == b);
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    return vec4(equal(v1, v2));
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslGreater(): GlslValueFunction {
  const name = 'greater_';
  const body = `
  float ${name}(float a, float b) {
    return float(a > b);
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    return vec4( v1.r > v2.r ,
      v1.g > v2.g,
      v1.b > v2.b,
      v1.a > v2.a );
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslLess(): GlslValueFunction {
  const name = 'less_';
  const body = `
  float ${name}(float a, float b) {
    return float(a < b);
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    return vec4( v1.r < v2.r ,
                v1.g < v2.g,
                v1.b < v2.b,
                v1.a < v2.a );
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslAnd(): GlslValueFunction {
  const name = 'and_';
  const body = `
  float ${name}(float a, float b) {
    return float( bool(a) && bool(b) );
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    bvec4 b1 = bvec4(v1);
    bvec4 b2 = bvec4(v2);
    return vec4( b1.r && b2.r ,
                b1.g && b2.g,
                b1.b && b2.b,
                b1.a && b2.a );
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslOr(): GlslValueFunction {
  const name = 'or_';
  const body = `
  float ${name}(float a, float b) {
    return float( bool(a) || bool(b) );
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    bvec4 b1 = bvec4(v1);
    bvec4 b2 = bvec4(v2);
    return vec4( b1.r || b2.r ,
                b1.g || b2.g,
                b1.b || b2.b,
                b1.a || b2.a );
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslXor(): GlslValueFunction {
  const name = 'xor_';
  const body = `
  float ${name}(float a, float b) {
    return float( bool(a) ^^ bool(b) );
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    bvec4 b1 = bvec4(v1);
    bvec4 b2 = bvec4(v2);
    return vec4( b1.r ^^ b2.r ,
                b1.g ^^ b2.g,
                b1.b ^^ b2.b,
                b1.a ^^ b2.a );
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslPow(): GlslValueFunction {
  return glslBuiltinBinary('pow');
}
export function glslPRelu(): GlslValueFunction {
  const name = 'prelu_';
  const body = `
  float ${name}(float a, float b) {
    return a < 0.0 ? a * b: a;
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    return vec4(
      v1.r < 0.0 ? v1.r * v2.r: v1.r,
      v1.g < 0.0 ? v1.g * v2.g: v1.g,
      v1.b < 0.0 ? v1.b * v2.b: v1.b,
      v1.a < 0.0 ? v1.a * v2.a: v1.a
      );
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}

function glslBuiltinBinary(fname: string): GlslValueFunction {
  const name = `${fname}_`;
  const body = `
  float ${name}(float a, float b) {
    return ${fname}(a, b);
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    return ${fname}(v1, v2);
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
