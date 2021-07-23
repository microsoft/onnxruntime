// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Graph} from '../../../graph';
import {OperatorImplementation, OperatorInitialization} from '../../../operators';
import {Tensor} from '../../../tensor';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, TextureType} from '../types';

export const leakyRelu: OperatorImplementation<number> =
    (inferenceHandler: WebGLInferenceHandler, inputs: Tensor[], alpha: number): Tensor[] => {
      validateInputs(inputs);
      const output = inferenceHandler.run(createLeakyReluProgramInfo(inferenceHandler, inputs, alpha), inputs);
      return [output];
    };

export const parseLeakyReluAttributes: OperatorInitialization<number> = (node: Graph.Node): number =>
    node.attributes.getFloat('alpha', 0.01);

const createLeakyReluProgramInfo = (handler: WebGLInferenceHandler, inputs: Tensor[], alpha: number): ProgramInfo => {
  const outputShape = inputs[0].dims.slice();
  const glsl = getGlsl(handler.session.backend.glContext.version);
  const shaderSource = `
      void main() {
        float v = ${glsl.texture2D}(A, TexCoords).r;
        ${glsl.output} = vec4(v < 0.0 ? v * float(${alpha}) : v);
      }`;
  return {
    name: 'LeakyRelu',
    inputNames: ['A'],
    inputTypes: [TextureType.unpacked],
    output: {dims: outputShape, type: inputs[0].type, textureType: TextureType.unpacked},
    hasMain: true,
    shaderSource
  };
};

const validateInputs = (inputs: Tensor[]): void => {
  if (!inputs || inputs.length !== 1) {
    throw new Error('LeakyRelu requires 1 input.');
  }
  if (inputs[0].type !== 'float32' && inputs[0].type !== 'float64') {
    throw new Error('Invalid input type.');
  }
};