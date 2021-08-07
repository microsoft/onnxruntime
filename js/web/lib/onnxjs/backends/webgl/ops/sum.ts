// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Sum} from '../../../ops/sum';
import {Tensor} from '../../../tensor';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

export class WebGLSum extends Sum implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const glsl = getGlsl(handler.session.backend.glContext.version);
    const outputShape = inputs[0].dims.slice();
    const sumLine = inputs.map((v, i) => `${glsl.texture2D}(X${i},TexCoords)`).join(' + ');
    const samplers = inputs.map((v, i) => `X${i}`);
    return {
      inputLayouts: inputs.map(t => handler.getOrCreateTextureLayout(t)),
      outputLayout: handler.createTextureLayoutFromShape(outputShape),
      samplers,
      shaderSource: `
      void main() {
        vec4 result = ${sumLine};
        ${glsl.output} = result;
      }`,
      hasMain: true
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = inputs.map((t, i) => handler.getOrCreateTextureData(t, programInfo.inputLayouts[i]));
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}
