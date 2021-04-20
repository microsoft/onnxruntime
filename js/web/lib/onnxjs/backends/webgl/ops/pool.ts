// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {AveragePool, GlobalAveragePool, GlobalMaxPool, MaxPool} from '../../../ops/pool';
import {Tensor} from '../../../tensor';
import {PoolConvUtil, ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, TextureLayout, WebGLOperator} from '../types';

export class WebGLGlobalAveragePool extends GlobalAveragePool implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    return createAveragePoolProgramInfo(
        inferenceHandler, inputs, true, this.kernelShape, this.autoPad, this.strides, this.pads, this.countIncludePad);
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

export class WebGLAveragePool extends AveragePool implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    return createAveragePoolProgramInfo(
        inferenceHandler, inputs, false, this.kernelShape, this.autoPad, this.strides, this.pads, this.countIncludePad);
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
function createAveragePoolProgramInfo(
    inferenceHandler: WebGLInferenceHandler, inputs: Tensor[], isGlobalOperator: boolean, kernelShape: number[] = [],
    autoPad = '', strides: number[] = [], pads: number[] = [], countIncludePad: boolean): ProgramInfo {
  const inputShape = inputs[0].dims.slice();
  PoolConvUtil.adjustPoolAttributes(isGlobalOperator, inputShape, kernelShape, strides, pads);
  const outputShape =
      PoolConvUtil.computePoolOutputShape(isGlobalOperator, inputShape, strides, kernelShape, pads, autoPad);
  const kernelSize = ShapeUtil.size(kernelShape);
  const op1 = 'value += _X(x);';
  let op2 = '';
  if (countIncludePad) {
    op2 += `value /= float(${kernelSize});`;
  } else {
    op2 += `value /= float(${kernelSize} - pad);`;
  }
  const inputLayout = inferenceHandler.getOrCreateTextureLayout(inputs[0]);
  const poolingCode = generatePoolingCode(inputLayout, kernelShape, pads, strides, op1, op2, '0.0');
  const shaderSource = `
      ${poolingCode}
    `;
  return {
    inputLayouts: [inputLayout],
    outputLayout: inferenceHandler.createTextureLayoutFromShape(outputShape),
    samplers: ['X'],
    shaderSource,
  };
}

export class WebGLGlobalMaxPool extends GlobalMaxPool implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    return createMaxPoolProgramInfo(
        inferenceHandler, inputs, true, this.kernelShape, this.autoPad, this.strides, this.pads);
  }
  createRunData(inferenceHandler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = [inferenceHandler.getOrCreateTextureData(inputs[0])];
    return {
      inputTextureDatas: inputTDs,
      outputTextureData:
          inferenceHandler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}

export class WebGLMaxPool extends MaxPool implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    return createMaxPoolProgramInfo(
        inferenceHandler, inputs, false, this.kernelShape, this.autoPad, this.strides, this.pads);
  }
  createRunData(inferenceHandler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = [inferenceHandler.getOrCreateTextureData(inputs[0])];
    return {
      inputTextureDatas: inputTDs,
      outputTextureData:
          inferenceHandler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}
function createMaxPoolProgramInfo(
    inferenceHandler: WebGLInferenceHandler, inputs: Tensor[], isGlobalOperator: boolean, kernelShape: number[] = [],
    autoPad = '', strides: number[] = [], pads: number[] = []): ProgramInfo {
  const inputShape = inputs[0].dims.slice();
  PoolConvUtil.adjustPoolAttributes(isGlobalOperator, inputShape, kernelShape, strides, pads);
  const outputShape =
      PoolConvUtil.computePoolOutputShape(isGlobalOperator, inputShape, strides, kernelShape, pads, autoPad);
  const op1 = `
              value = max(_X(x), value);
      `;
  const op2 = '';
  const inputLayout = inferenceHandler.createTextureLayoutFromShape(inputShape);
  const poolingCode = generatePoolingCode(inputLayout, kernelShape, pads, strides, op1, op2, '-1e5');
  const shaderSource = `
    ${poolingCode}
  `;
  return {
    inputLayouts: [inputLayout],
    outputLayout: inferenceHandler.createTextureLayoutFromShape(outputShape),
    samplers: ['X'],
    shaderSource,
  };
}

export function generatePoolingCode(
    x: TextureLayout, kernelShape: number[], pads: number[], strides: number[], op1: string, op2: string,
    startVal: string): string {
  const inputDims = x.shape;
  const rank = x.shape.length;
  if (kernelShape.length <= 2) {
    const kw = kernelShape[kernelShape.length - 1];
    const sw = strides[strides.length - 1];
    const pwStart = pads[pads.length / 2 - 1];
    const pwEnd = pads[pads.length - 1];
    const dimW = inputDims[rank - 1];
    let codeW = '';
    let codeH = '';
    let codeHEnd = '';
    if (pwStart + pwEnd !== 0) {
      codeW = `
                for (int i = 0; i < ${kw}; i++) {
                  x[${rank} - 1] = indices[${rank} - 1] * ${sw} - ${pwStart} + i;
                  if (x[${rank} - 1] < 0 || x[${rank} - 1] >= ${dimW}) {
                    pad++;
                    continue;
                  }
                  ${op1}
                }`;
    } else {
      codeW = `
                for (int i = 0; i < ${kw}; i++) {
                  x[${rank} - 1] = indices[${rank} - 1] * ${sw} - ${pwStart} + i;
                  ${op1}
                }`;
    }

    if (kernelShape.length === 2) {
      const kh = kernelShape[kernelShape.length - 2];
      const sh = strides[strides.length - 2];
      const phStart = pads[pads.length / 2 - 2];
      const phEnd = pads[pads.length - 2];
      const dimH = inputDims[rank - 2];
      if (phStart + phEnd !== 0) {
        codeH = `
              for (int j = 0; j < ${kh}; j++) {
                x[${rank} - 2] = indices[${rank} - 2] * ${sh} - ${phStart} + j;
                if (x[${rank} - 2] < 0 || x[${rank} - 2] >= ${dimH}) {
                  pad+= ${kw};
                  continue;
                }
            `;
      } else {
        codeH = `
                for (int j = 0; j < ${kh}; j++) {
                  x[${rank} - 2] = indices[${rank} - 2] * ${sh} - ${phStart} + j;
                `;
      }
      codeHEnd = `
              }
            `;
    }

    const poolingCode = `
            float process(int indices[${rank}]) {
              int x[${rank}];
              copyVec(indices, x);

              float value = ${startVal};
              int pad = 0;
              ${codeH}
              ${codeW}
              ${codeHEnd}
              ${op2}
              return value;
            }
          `;
    return poolingCode;
  } else {
    const kernelSize = ShapeUtil.size(kernelShape);
    const kernelStrides = ShapeUtil.computeStrides(kernelShape);
    const stridesRank = kernelStrides.length;
    const padsRank = pads.length;
    const offsetToIndicesFunction = offsetToIndices(stridesRank);
    const copyInputDims = copyArray(inputDims, 'inputDims');
    const copyPads = copyArray(pads, 'pads');
    const copyKernelStrides = copyArray(kernelStrides, 'kernelStrides');
    const copyStrides = copyArray(strides, 'strides');
    const hasPads = pads.reduce((sum, cur) => sum + cur);
    let padCode = '';
    if (hasPads) {
      padCode = `
                if (x[j] >= inputDims[j] || x[j] < 0) {
                  pad++;
                  isPad = true;
                  break;
                }
              }
              if (!isPad) {
                ${op1}
              }`;
    } else {
      padCode = `
                  }
                  ${op1}`;
    }
    const poolingCode = `
            ${offsetToIndicesFunction}
            float process(int indices[${rank}]) {
                int x[${rank}];
                copyVec(indices, x);
                int offset[${stridesRank}];
                int pads[${padsRank}];
                int inputDims[${rank}];
                int kernelStrides[${stridesRank}];
                int strides[${stridesRank}];
                ${copyPads}
                ${copyInputDims}
                ${copyStrides}
                ${copyKernelStrides}

                float value = ${startVal};
                int pad = 0;
                bool isPad = false;
                for (int i = 0; i < ${kernelSize}; i++) {
                    offsetToIndices(i, kernelStrides, offset);
                    isPad = false;
                    for (int j = ${rank} - ${stridesRank}; j < ${rank}; j++) {
                      x[j] = indices[j] * strides[j - ${rank} + ${stridesRank}]
                        + offset[j - ${rank} + ${stridesRank}] - pads[j - 2];
                      ${padCode}
                }
                ${op2}

                return value;
            }`;
    return poolingCode;
  }
}

export function copyArray(array: readonly number[], arrayName: string): string {
  let block = '';
  for (let i = 0; i < array.length; i++) {
    block += `
      ${arrayName}[${i}] = ${array[i]};
    `;
  }
  return block;
}

export function offsetToIndices(rank: number): string {
  return `
    void offsetToIndices(int offset, int[${rank}] strides, out int[${rank}] indices) {
      if (${rank} == 0) {
        return;
      }
      for (int i = 0; i < ${rank} - 1; ++i) {
        indices[i] = offset / strides[i];
        offset -= indices[i] * strides[i];
      }
      indices[${rank} - 1] = offset;
    }`;
}
