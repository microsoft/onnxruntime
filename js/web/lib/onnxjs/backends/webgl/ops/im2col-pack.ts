// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from '../../../tensor';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, TextureType} from '../types';
import {ConvAttributes} from './conv';
import {unpackFromChannel} from './packing-utils';

export const createPackedIm2ColProgramInfo =
    (inferenceHandler: WebGLInferenceHandler, x: Tensor, w: Tensor, outputShape: readonly number[],
     attributes: ConvAttributes): ProgramInfo => {
      const xshape = x.dims;
      const wshape = w.dims;
      const rowDim = 2;
      const colDim = 3;
      const rank = outputShape.length;
      const im2colShape = [wshape[1] * wshape[2] * wshape[3], outputShape[2] * outputShape[3]];
      const kernelSize = wshape[2] * wshape[3];
      const unpackChannel = unpackFromChannel();
      const glsl = getGlsl(inferenceHandler.session.backend.glContext.version);
      let unrolled = '';

      for (let row = 0; row <= 1; row++) {
        for (let col = 0; col <= 1; col++) {
          unrolled += `
            blockIndex = rc.x + ${col};
            pos = rc.y + ${row};

            if(blockIndex < ${im2colShape[1]} && pos < ${im2colShape[0]}) {
              offsetY = int(blockIndex / (${outputShape[rank - 1]})) * ${attributes.strides[0]} - ${
              attributes.pads[0]};
              d0 = offsetY + ${attributes.dilations[0]} * (imod(pos, ${kernelSize}) / ${wshape[2]});

              if(d0 < ${xshape[rowDim]} && d0 >= 0) {
                offsetX = imod(blockIndex, ${outputShape[rank - 1]}) * ${attributes.strides[1]} - ${
              attributes.pads[1]};
                d1 = offsetX + ${attributes.dilations[1]} * imod(imod(pos, ${kernelSize}), ${wshape[2]});

                if(d1 < ${xshape[colDim]} && d1 >= 0) {

                  ch = int(float(pos)/ ${kernelSize}.);
                    innerDims = vec2(d0, d1);
                    result[${row * 2 + col}] = getChannel(
                      getA(0, ch, int(innerDims.x),
                      int(innerDims.y)), innerDims);
                }
              }
            }

          `;
        }
      }

      const shaderSource = `
      ${unpackChannel}

      void main() {
        ivec2 rc = getOutputCoords();
          vec4 result = vec4(0.0);
          int blockIndex, pos, offsetY, d0, offsetX, d1, ch;
          vec2 innerDims;
          ${unrolled}
          ${glsl.output} = result;
      }
            `;
      return {
        name: 'Im2Col (packed)',
        inputNames: ['A'],
        inputTypes: [TextureType.packed],
        output: {dims: im2colShape, type: x.type, textureType: TextureType.packed},
        shaderSource,
        hasMain: true
      };
    };
