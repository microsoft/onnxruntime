// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Gather} from '../../../ops/gather';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

export class WebGLGather extends Gather implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }

  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const inputShape = inputs[0].dims.slice();
    const indexDataShape = inputs[1].dims.slice();
    const outputShape = new Array(inputShape.length + indexDataShape.length - 1);

    const axis = ShapeUtil.normalizeAxis(this.axis, inputShape.length);
    const indexCopyOps: string[] = [];
    for (let i = 0; i < outputShape.length; i++) {
      // outputShape is divided into three parts: A, B, C
      // |0        axis|  axis + indexDataShape.length |          end|
      // |     A       |             B                 |      C      |
      //
      // inputIdx: [A, inputs[1][B], C]
      if (i < axis) {  // A
        outputShape[i] = inputShape[i];
        indexCopyOps.push(`inputIdx[${i}] = outputIdx[${i}];`);
      } else {
        if (i < axis + indexDataShape.length) {  // B
          outputShape[i] = indexDataShape[i - axis];
          indexCopyOps.push(`indexDataIdx[${i - axis}] = outputIdx[${i}];`);
        } else {                                                       // C
          outputShape[i] = inputShape[i - indexDataShape.length + 1];  // skip 1 for axis
          indexCopyOps.push(`inputIdx[${i - indexDataShape.length + 1}] = outputIdx[${i}];`);
        }
      }
    }

    const orank = outputShape.length || 1;
    const irank = inputShape.length;
    const iDrank = indexDataShape.length || 1;
    const shaderSource = `
      float process(int outputIdx[${orank}]) {
        int inputIdx[${irank}];
        int indexDataIdx[${iDrank}];
        indexDataIdx[0] = 0;
        ${indexCopyOps.join('\n        ')}
        int idx = int(_B(indexDataIdx));
        inputIdx[${axis}] = idx < 0 ? idx + ${inputShape[axis]} : idx;
        return _A(inputIdx);
      }`;
    return {
      inputLayouts: inputs.map(t => handler.getOrCreateTextureLayout(t)),
      outputLayout: handler.createTextureLayoutFromShape(outputShape),
      samplers: ['A', 'B'],
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
