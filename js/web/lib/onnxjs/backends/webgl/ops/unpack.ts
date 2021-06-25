// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from '../../../tensor';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, TextureType} from '../types';
import {getCoordsDataType} from '../utils';
import {getChannels, unpackFromChannel} from './packing-utils';

export const creatUnpackProgramInfo = (handler: WebGLInferenceHandler, input: Tensor): ProgramInfo => {
  const rank = input.dims.length;

  const channels = getChannels('rc', rank);
  const innerDims = channels.slice(-2);
  const coordsDataType = getCoordsDataType(rank);
  const unpackChannel = unpackFromChannel();
  const isScalar = (input.dims.length === 0);
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
    inputNames: ['A'],
    inputTypes: [TextureType.unpacked],
    output: {dims: input.dims, type: input.type, textureType: TextureType.packed},
    shaderSource
  };
};

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
