// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from '../../../tensor';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, TextureType} from '../types';

import {createPackedReshapeProgramInfo} from './reshape-packed';

const createReshapeProgramInfo = (handler: WebGLInferenceHandler, input0: Tensor, input1: Tensor): ProgramInfo => {
  if (handler.session.pack) {
    return createPackedReshapeProgramInfo(handler, input0, input1);
  } else {
    // TODO: how do we handle no-op? For reshape in unpacked mode, we don't need to run
    // any shaders. Instead, we can just use the same tensor data with different dims value.
    // This would require a change in how we represent tensor and whether we allow de-couple
    // tensor data (memory) with its meta-data (like dims, types ect).
    // Before we implement the feature above, temporarily return a dummy programInfo.
    const glsl = getGlsl(handler.session.backend.glContext.version);
    return {
      inputTypes: [TextureType.unpacked],
      inputNames: ['A'],
      output: {dims: Array.from(input1.integerData), type: input0.type, textureType: TextureType.unpacked},
      shaderSource: `
             void main() {
               vec4 v = ${glsl.texture2D}(A, TexCoords);
               ${glsl.output} = v;
             }
             `,
      hasMain: true
    };
  }
};

export const reshape = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createReshapeProgramInfo(handler, inputs[0], inputs[1]), inputs)];
