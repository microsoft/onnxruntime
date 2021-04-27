// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {ReduceBase} from '../../../ops/reduce-op';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

abstract class WebGLGenericReduce extends ReduceBase implements WebGLOperator {
  abstract getOps(inputs: Tensor[], axes: number[]): string[];

  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const outputShape: number[] = [];
    const iRank = inputs[0].dims.length || 1;

    const idxCopy = [];  // copy output indexes to input indexes

    const axes = ShapeUtil.normalizeAxes(this.axes, inputs[0].dims.length);
    const ops = this.getOps(inputs, axes);  // [init ops, reduce ops, final ops]
    let reduceOps = ops[1];

    for (let k = 0; k < inputs[0].dims.length; k++) {
      // if this axis is reduced
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        if (this.keepDims) {
          outputShape.push(1);
        }  // else { remove the axis from outputShape; }

        // loop over the d-th axis
        reduceOps = `
        for(int j${k} = 0; j${k} < ${inputs[0].dims[k]}; j${k}++) {
          inputIdx[${k}] = j${k};
          ${reduceOps}
        }
        `;
      } else {
        idxCopy.push(`inputIdx[${k}] = outputIdx[${outputShape.length}];`);

        outputShape.push(inputs[0].dims[k]);
      }
    }

    const oRank = outputShape.length || 1;

    const shaderSource = `
      float process(int outputIdx[${oRank}]) {
        float value;                 // final result
        int inputIdx[${iRank}];      // addressing input data
        ${idxCopy.join('\n')}
        ${ops[0]}       // init ops for reduce max/min
        ${reduceOps}
        ${ops[2]}       // final computation for reduce mean
        return value;
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

export class WebGLReduceSum extends WebGLGenericReduce {
  getOps(_inputs: Tensor[]): string[] {
    return ['value = 0.0;', 'value += _A(inputIdx);', ''];
  }
}

export class WebGLReduceMean extends WebGLGenericReduce {
  getOps(inputs: Tensor[], axes: number[]): string[] {
    let size = 1.0;
    for (let k = 0; k < inputs[0].dims.length; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        size *= inputs[0].dims[k];
      }
    }

    return ['value = 0.0;', 'value += _A(inputIdx);', `value /= ${size}.;`];  // ensure real number with `.`
  }
}

export class WebGLReduceMax extends WebGLGenericReduce {
  getOps(inputs: Tensor[], axes: number[]): string[] {
    const idxZero = [];
    for (let k = 0; k < inputs[0].dims.length; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        idxZero.push(`inputIdx[${k}] = 0;`);  // first element
      }
    }

    return [`${idxZero.join('\n')}\nvalue = _A(inputIdx);`, 'value = max(value, _A(inputIdx));', ''];
  }
}

export class WebGLReduceMin extends WebGLGenericReduce {
  getOps(inputs: Tensor[], axes: number[]): string[] {
    const idxZero = [];
    for (let k = 0; k < inputs[0].dims.length; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        idxZero.push(`inputIdx[${k}] = 0;`);  // first element
      }
    }

    return [`${idxZero.join('\n')}\nvalue = _A(inputIdx);`, 'value = min(value, _A(inputIdx));', ''];
  }
}

export class WebGLReduceProd extends WebGLGenericReduce {
  getOps(_inputs: Tensor[]): string[] {
    return ['value = 1.0;', 'value *= _A(inputIdx);', ''];
  }
}

export class WebGLReduceLogSum extends WebGLGenericReduce {
  getOps(_inputs: Tensor[]): string[] {
    return ['value = 0.0;', 'value += _A(inputIdx);', 'value = log(value);'];
  }
}

export class WebGLReduceSumSquare extends WebGLGenericReduce {
  getOps(_inputs: Tensor[]): string[] {
    return ['float t; value = 0.0;', 't = _A(inputIdx); value += t * t;', ''];
  }
}
