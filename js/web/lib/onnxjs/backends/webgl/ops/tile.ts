// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tile} from '../../../ops/tile';
import {Tensor} from '../../../tensor';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

export class WebGLTile extends Tile implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }

  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const inputShape = inputs[0].dims.slice();
    const outputShape = new Array(inputShape.length);  // inputs[0].dims.slice();

    const tileOps: string[] = [];
    for (let i = 0; i < inputShape.length; i++) {
      outputShape[i] = inputShape[i] * inputs[1].numberData[i];
      tileOps.push(`inputIdx[${i}] = int(mod(float(outputIdx[${i}]), ${inputShape[i]}.));`);
    }

    const rank = outputShape.length;
    const shaderSource = `
    float process(int outputIdx[${rank}]) {
      int inputIdx[${rank}];
      ${tileOps.join('\n')}
      return _A(inputIdx);
    }`;
    return {
      inputLayouts: inputs.map(t => handler.getOrCreateTextureLayout(t)),
      outputLayout: handler.createTextureLayoutFromShape(outputShape),
      samplers: ['A'],
      shaderSource,
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = inputs.map((t, i) => handler.getOrCreateTextureData(t, programInfo.inputLayouts[i]));
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}
