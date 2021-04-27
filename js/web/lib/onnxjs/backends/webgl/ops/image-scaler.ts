// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {ImageScaler} from '../../../ops/image-scaler';
import {Tensor} from '../../../tensor';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

export class WebGLImageScaler extends ImageScaler implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const outputShape = inputs[0].dims.slice();
    const rank = outputShape.length;
    const getBiasMethod = this.createGetBiasMethod(this.bias.length);
    const shaderSource = `
      ${getBiasMethod}
      float process(int indices[${rank}]) {
        return _X(indices) * scale + getBias(bias, indices[1]);
      }`;
    return {
      inputLayouts: [handler.getOrCreateTextureLayout(inputs[0])],
      outputLayout: handler.createTextureLayoutFromShape(outputShape),
      samplers: ['X'],
      variables: [{name: 'bias', type: 'float', arrayLength: this.bias.length}, {name: 'scale', type: 'float'}],
      shaderSource,
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = [handler.getOrCreateTextureData(inputs[0], programInfo.inputLayouts[0])];
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {'bias': this.bias, 'scale': this.scale}
    };
  }
  private createGetBiasMethod(numChannels: number): string {
    const codeLines: string[] = [`float getBias(float bias[${numChannels}], int channel) {`];
    for (let i = 0; i < numChannels; ++i) {
      if (i === 0) {
        codeLines.push(
            '\t' +
            `if (channel == ${i}) { return bias[${i}]; }`);
      } else if (i === numChannels - 1) {
        codeLines.push(
            '\t' +
            `else { return bias[${i}]; }`);
      } else {
        codeLines.push(
            '\t' +
            `else if (channel == ${i}) { return bias[${i}]; }`);
      }
    }
    codeLines.push(
        '\t' +
        '}');
    return codeLines.join('\n');
  }
}
