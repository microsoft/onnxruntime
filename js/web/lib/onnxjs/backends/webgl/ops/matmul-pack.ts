// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {MatMul} from '../../../ops/matmul';
import {Tensor} from '../../../tensor';
import {BroadcastUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';
import {getActicationSnippet} from './fuse_utils';

export class WebGLMatMulPacked extends MatMul implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const hasBias = inputs.length > 2;
    const processBias = hasBias ? 'value += vec4(getBias(a[0]*2).xx, getBias(a[0]*2).yy);' : '';
    const aShape = inputs[0].dims;
    const bShape = inputs[1].dims;
    const outputShape = BroadcastUtil.calcShape(aShape, bShape, true);

    if (!outputShape) {
      throw new Error('Can\'t use matmul on the given tensors');
    }
    const rank = outputShape.length;
    const aRank = aShape.length;
    const bRank = bShape.length;
    const sharedDim = aShape[aShape.length - 1];

    const {activationFunction, applyActivation} = getActicationSnippet(this.activation);
    // TODO:fix broadcasting
    const shaderSource = `
      ${activationFunction}
      vec4 process(int indices[${rank}]) {
          int a[${aRank}];
          int b[${bRank}];
          bcastMatmulIndices_A(indices, a);
          bcastMatmulIndices_B(indices, b);

          vec4 value;
          for (int k=0; k<((${sharedDim}+1)/2); ++k) {
              a[${aRank - 1}] = k;
              b[${bRank - 2}] = k;
              value += ${getA(aRank)}.rrbb * ${getB(bRank)}.rgrg;
              value += ${getA(aRank)}.ggaa * ${getB(bRank)}.baba;
          }
          ${processBias}
          ${applyActivation}
          return value;
      }`;
    return {
      inputLayouts: inputs.map((t, i) => handler.getOrCreateTextureLayout(t, 4, true, inputs[i].dims, true)),
      outputLayout:
          handler.createTextureLayoutFromShape(outputShape, 4, outputShape, {isPacked: true, reverseWH: true}),
      samplers: hasBias ? ['A', 'B', 'Bias'] : ['A', 'B'],
      shaderSource,
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

function getA(outputRank: number): string {
  let res = 'getA(';
  for (let i = 0; i < outputRank - 2; i++) {
    res += `a[${i}], `;
  }
  res += `a[${outputRank - 2}]*2, ` +
      'k*2)';
  return res;
}

function getB(outputRank: number): string {
  let res = 'getB(';
  for (let i = 0; i < outputRank - 2; i++) {
    res += `b[${i}], `;
  }
  res += 'k*2, ' +
      `b[${outputRank - 1}]*2)`;
  return res;
}
