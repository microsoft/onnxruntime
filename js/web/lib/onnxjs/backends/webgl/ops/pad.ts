// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Pad} from '../../../ops/pad';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {getGlsl, Glsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, TextureLayout, WebGLOperator} from '../types';

export class WebGLPad extends Pad implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const outputShape = ShapeUtil.padShape(inputs[0].dims.slice(), this.pads);
    const rank = outputShape.length;
    const alayout = inferenceHandler.getOrCreateTextureLayout(inputs[0]);
    const padFunction = getPadFunction(
        getGlsl(inferenceHandler.session.backend.glContext.version), 'A', alayout, this.mode, this.pads, this.value);
    const shaderSource = `
      ${padFunction}
      float process(int[${rank}] indices) {
          return padA(indices);
      }`;
    return {
      inputLayouts: [alayout],
      outputLayout: inferenceHandler.createTextureLayoutFromShape(outputShape),
      samplers: ['A'],
      shaderSource,
    };
  }
  createRunData(inferenceHandler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = [inferenceHandler.getOrCreateTextureData(inputs[0], programInfo.inputLayouts[0])];
    return {
      inputTextureDatas: inputTDs,
      outputTextureData:
          inferenceHandler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}
export function getPadFunction(
    glsl: Glsl, name: string, inputLayout: TextureLayout, mode: string, pads: number[], value: number): string {
  switch (mode) {
    case 'constant':
      return getPadConstant(
          glsl, name, inputLayout.shape, inputLayout.strides, inputLayout.width, inputLayout.height, pads, value);
    case 'reflect':
      return getPadReflect(
          glsl, name, inputLayout.shape, inputLayout.strides, inputLayout.width, inputLayout.height, pads);
    case 'edge':
      return getPadEdge(
          glsl, name, inputLayout.shape, inputLayout.strides, inputLayout.width, inputLayout.height, pads);
    default:
      throw new Error('Invalid mode');
  }
}
function getPadConstant(
    glsl: Glsl, name: string, shape: readonly number[], strides: readonly number[], width: number, height: number,
    pads: number[], value: number) {
  const rank = shape.length;
  let block = '';
  for (let i = rank - 1; i >= 0; --i) {
    block += `
          k = m[${i}] - ${pads[i]};
          if (k < 0)  return constant;
          if (k >= ${shape[i]}) return constant;
          offset += k * ${strides[i]};
          `;
  }
  return `
        float pad${name}(int m[${rank}]) {
          const float constant = float(${value});
          int offset = 0;
          int k = 0;
          ${block}
          vec2 coords = offsetToCoords(offset, ${width}, ${height});
          float value = getColorAsFloat(${glsl.texture2D}(${name}, coords));
          return value;
        }
        `;
}
function getPadReflect(
    glsl: Glsl, name: string, shape: readonly number[], strides: readonly number[], width: number, height: number,
    pads: number[]) {
  const rank = shape.length;

  let block = '';
  for (let i = rank - 1; i >= 0; --i) {
    block += `
        k = m[${i}] - ${pads[i]};
        if (k < 0) { k = -k; }
        {
          const int _2n_1 = ${2 * (shape[i] - 1)};
          k = int( mod( float(k), float(_2n_1) ) ) ;
          if(k >= ${shape[i]}) { k = _2n_1 - k; }
        }
        offset += k * ${strides[i]};
        `;
  }
  return `
      float pad${name}(int m[${rank}]) {
        int offset = 0;
        int k = 0;
        ${block}
        vec2 coords = offsetToCoords(offset, ${width}, ${height});
        float value = getColorAsFloat(${glsl.texture2D}(${name}, coords));
        return value;
      }
      `;
}
function getPadEdge(
    glsl: Glsl, name: string, shape: readonly number[], strides: readonly number[], width: number, height: number,
    pads: number[]) {
  const rank = shape.length;

  let block = '';
  for (let i = rank - 1; i >= 0; --i) {
    block += `
      k = m[${i}] - ${pads[i]};
      if (k < 0)  k = 0;
      if (k >= ${shape[i]}) k = ${shape[i] - 1};
      offset += k * ${strides[i]};
      `;
  }
  return `
    float pad${name}(int m[${rank}]) {
      int offset = 0;
      int k = 0;
      ${block}
      vec2 coords = offsetToCoords(offset, ${width}, ${height});
      float value = getColorAsFloat(${glsl.texture2D}(${name}, coords));
      return value;
    }
    `;
}
