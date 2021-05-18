// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from '../../../tensor';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';
import {unpackFromChannel} from './packing-utils';

export class WebGLIm2ColPacked implements WebGLOperator {
  protected convOutputShape: number[];
  protected kernelShape: number[];
  protected dilations: number[];
  protected pads: number[];
  protected strides: number[];

  constructor(
      convOutputShape: number[], kernelShape: number[], dilations: number[], pads: number[], strides: number[]) {
    this.convOutputShape = convOutputShape;
    this.kernelShape = kernelShape;
    this.dilations = dilations;
    this.pads = pads;
    this.strides = strides;
  }

  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    if (inputs.length !== 2) {
      throw new Error('Im2Col kernel should have two input tensors');
    }

    const xshape = inputs[0].dims.slice();
    const wshape = inputs[1].dims.slice();
    const rowDim = 2;
    const colDim = 3;
    const rank = this.convOutputShape.length;
    const im2colShape = [wshape[1] * wshape[2] * wshape[3], this.convOutputShape[2] * this.convOutputShape[3]];
    const kernelSize = wshape[2] * wshape[3];
    const unpackChannel = unpackFromChannel();
    let unrolled = '';

    for (let row = 0; row <= 1; row++) {
      for (let col = 0; col <= 1; col++) {
        unrolled += `
          blockIndex = rc.x + ${col};
          pos = rc.y + ${row};

          if(blockIndex < ${im2colShape[1]} && pos < ${im2colShape[0]}) {
            offsetY = int(blockIndex / (${this.convOutputShape[rank - 1]})) * ${this.strides[0]} - ${this.pads[1]};
            d0 = offsetY + ${this.dilations[0]} * (imod(pos, ${kernelSize}) / ${wshape[2]});

            if(d0 < ${xshape[rowDim]} && d0 >= 0) {
              offsetX = imod(blockIndex, ${this.convOutputShape[rank - 1]}) * ${this.strides[1]} - ${this.pads[0]};
              d1 = offsetX + ${this.dilations[1]} * imod(imod(pos, ${kernelSize}), ${wshape[2]});

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
        outputColor = result;
    }
          `;
    return {
      inputLayouts: [inferenceHandler.getOrCreateTextureLayout(inputs[0], 4, true, xshape, true)],
      outputLayout:
          inferenceHandler.createTextureLayoutFromShape(im2colShape, 4, im2colShape, {isPacked: true, reverseWH: true}),
      samplers: ['A'],
      shaderSource,
      hasMain: true,
      expectPackedInputs: true,
      expectPackedOutputs: true,
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs =
        inputs.map((t) => handler.getOrCreateTextureData(t, handler.getOrCreateTextureLayout(t, 1, false, [], true)));
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}
