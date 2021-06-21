// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* eslint-disable no-invalid-this */

import {Clip} from '../../../ops/clip';
import {Tensor} from '../../../tensor';
import {FunctionType} from '../glsl-definitions';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

export class WebGLClip extends Clip implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const outputShape = inputs[0].dims.slice();
    const glsl = getGlsl(handler.session.backend.glContext.version);
    const shaderSource = `
      const float min = float(${this.min});
      const float max = float(${this.max});
      void main() {
        float v = ${glsl.texture2D}(A, TexCoords).r;
        ${glsl.output} = vec4(clamp(v, min, max));
      }
      `;
    return {
      inputLayouts: [handler.getOrCreateTextureLayout(inputs[0])],
      outputLayout: handler.createTextureLayoutFromShape(outputShape),
      samplers: ['A'],
      shaderSource,
      hasMain: true,
    };
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
// used for fusion
export function glslClip() {
  const name = 'clip_';
  const body = `
    float ${name}(float a, float max, float min) {
      return clamp(a, min, max);
    }
    vec4 ${name}(vec4 v, float max, float min) {
      return clamp(v, min, max);
    }
    `;
  return {body, name, type: FunctionType.ValueBased};
}
