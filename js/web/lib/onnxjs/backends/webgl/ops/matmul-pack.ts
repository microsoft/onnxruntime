// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {MatMul} from '../../../ops/matmul';
import {Tensor} from '../../../tensor';
import {BroadcastUtil} from '../../../util';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';
import {getCoordsDataType} from '../utils';

import {getActicationSnippet} from './fuse-utils';

export class WebGLMatMulPacked extends MatMul implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const hasBias = inputs.length > 2;
    const processBias = hasBias ? 'result += getBiasAtOutCoords();' : '';
    const aShape = inputs[0].dims;
    const bShape = inputs[1].dims;
    const outputShape = BroadcastUtil.calcShape(aShape, bShape, true);

    if (!outputShape) {
      throw new Error('Can\'t use matmul on the given tensors');
    }
    const sharedDim = aShape[aShape.length - 1];
    const sharedDimIndex = Math.ceil(sharedDim / 2);
    const aRank = aShape.length;
    const bRank = bShape.length;

    const glsl = getGlsl(handler.session.backend.glContext.version);
    const coordsDataType = getCoordsDataType(outputShape.length);
    const allGlChannels = ['x', 'y', 'z', 'w', 'u', 'v'];

    const {activationFunction, applyActivation} = getActicationSnippet(this.activation);
    // TODO:fix broadcasting
    const shaderSource = `
      ${activationFunction}
      void main() {
        ${coordsDataType} rc = getOutputCoords();
        vec4 result = vec4(0);
        for (int i = 0; i < ${sharedDimIndex}; i++) {
          vec4 a = getA(${getA(allGlChannels, aRank)});
          vec4 b = getB(${getB(allGlChannels, bRank)});
          result += (a.rrbb * b.rgrg);
          result += (a.ggaa * b.baba);
        }
        ${processBias}
        ${applyActivation}
        ${glsl.output} = result;
      }`;
    return {
      name: 'WebGLMatMulPacked',
      inputLayouts: inputs.map((t, i) => handler.getOrCreateTextureLayout(t, 4, true, inputs[i].dims, true)),
      outputLayout:
          handler.createTextureLayoutFromShape(outputShape, 4, outputShape, {isPacked: true, reverseWH: true}),
      samplers: hasBias ? ['A', 'B', 'Bias'] : ['A', 'B'],
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

function getA(allGlChannels: string[], rank: number): string {
  let res = '';
  for (let i = 0; i < rank - 2; i++) {
    res += `rc.${allGlChannels[i]}, `;
  }
  res += `rc.${allGlChannels[rank - 2]}, ` +
      'i<<1';
  return res;
}

function getB(allGlChannels: string[], rank: number): string {
  let res = '';
  for (let i = 0; i < rank - 2; i++) {
    res += `rc.${allGlChannels[i]}, `;
  }
  res += 'i<<1, ' +
      `rc.${allGlChannels[rank - 1]}`;
  return res;
}
