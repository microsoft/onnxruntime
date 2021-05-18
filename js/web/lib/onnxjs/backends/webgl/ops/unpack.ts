// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from '../../../tensor';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';
import {getCoordsDataType} from '../utils';
import {getChannels, unpackFromChannel} from './packing-utils';

export class WebGLUnpack implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    if (inputs.length !== 1) {
      throw new Error('Pack kernel should have input tensor count to 1.');
    }

    const inputTexture = handler.getTextureData(inputs[0].dataId, true);
    if (!inputTexture) {
      throw new Error('packed input texture must exist');
    }

    const inputLayout = handler.getOrCreateTextureLayout(inputs[0], 4, true);
    const isScalar = (inputLayout.unpackedShape.length === 0);
    const outputLayout = handler.createTextureLayoutFromShape(inputTexture.unpackedShape);
    const outputShape = outputLayout.shape;
    const rank = outputShape.length;

    const channels = getChannels('rc', rank);
    const innerDims = channels.slice(-2);
    const coordsDataType = getCoordsDataType(rank);
    const unpackChannel = unpackFromChannel();
    const sourceCoords = isScalar ? '' : getSourceCoords(rank, channels);
    const coords = rank <= 1 ? 'rc' : `vec2(${innerDims.join(',')})`;
    const glsl = getGlsl(handler.session.backend.glContext.version);
    const shaderSource = `
    ${unpackChannel}
    void main() {
      ${coordsDataType} rc = getOutputCoords();

      // Sample the texture with the coords to get the rgba channel value.
      vec4 packedInput = getA(${sourceCoords});

      ${glsl.output} = vec4(getChannel(packedInput, ${coords}), 0, 0, 0);
    }
  `;

    return {
      name: 'WebGLUnpack',
      inputLayouts: [handler.getOrCreateTextureLayout(inputs[0], 4, true, inputs[0].dims, true)],
      outputLayout,
      samplers: ['A'],
      shaderSource,
      hasMain: true,
      expectPackedInputs: true,
      expectPackedOutputs: false,
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = [handler.getOrCreateTextureData(inputs[0], programInfo.inputLayouts[0], true)];
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}

export function getSourceCoords(rank: number, dims: string[]): string {
  if (rank === 1) {
    return 'rc';
  }

  let coords = '';
  for (let i = 0; i < rank; i++) {
    coords += dims[i];
    if (i < rank - 1) {
      coords += ',';
    }
  }
  return coords;
}
