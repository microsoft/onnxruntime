// Licensed under the MIT license.

import {Slice, SliceV10} from '../../../ops/slice';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

export class WebGLSlice extends Slice implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }

  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    return createProgramInfo(handler, inputs[0], this.starts, this.ends, this.axes);
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    return createRunData(handler, programInfo, inputs);
  }
}

export class WebGLSliceV10 extends SliceV10 implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }

  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    if (!handler.session.isInitializer(inputs[1].dataId) || !handler.session.isInitializer(inputs[2].dataId) ||
        (inputs.length >= 4 && !handler.session.isInitializer(inputs[3].dataId)) ||
        (inputs.length >= 5 && !handler.session.isInitializer(inputs[4].dataId))) {
      throw new Error('dynamic slice attributes are not allowed');
    }
    if (inputs.length >= 5 && inputs[4].integerData.some((i: number) => i !== 1)) {
      throw new Error('currently non-1 steps is not supported for Slice');
    }
    const starts = Array.from(inputs[1].integerData);
    const ends = Array.from(inputs[2].integerData);
    const axes = inputs.length >= 4 ? Array.from(inputs[3].integerData) : [];

    return createProgramInfo(handler, inputs[0], starts, ends, axes);
  }

  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    return createRunData(handler, programInfo, inputs);
  }
}

function createProgramInfo(
    handler: WebGLInferenceHandler, x: Tensor, starts: readonly number[], ends: readonly number[],
    axes: readonly number[]): ProgramInfo {
  if (axes.length === 0) {
    axes = x.dims.slice(0).map((val, ind) => ind);
  }
  axes = ShapeUtil.normalizeAxes(axes, x.dims.length);
  starts = starts.map((start, ind) => {
    if (start > x.dims[axes[ind]] - 1) {
      return x.dims[axes[ind]];
    }
    return ShapeUtil.normalizeAxis(start, x.dims[axes[ind]]);
  });
  ends = ends.map((end, ind) => {
    if (end > x.dims[axes[ind]] - 1) {
      return x.dims[axes[ind]];
    }
    return ShapeUtil.normalizeAxis(end, x.dims[axes[ind]]);
  });

  const outputShape = x.dims.slice();

  const sliceOps: string[] = [];
  for (let i = 0; i < axes.length; i++) {
    outputShape[axes[i]] = ends[i] - starts[i];
    if (starts[i] > 0) {
      sliceOps.push(`outputIdx[${axes[i]}] += ${starts[i]};`);
    }  // else { sliceOps.push(`outputIdx[${axes[i]}] += 0;`); }
  }

  const rank = outputShape.length;
  const shaderSource = `
      float process(int outputIdx[${rank}]) {
        ${sliceOps.join('\n      ')}
        return _A(outputIdx);
      }`;
  return {
    inputLayouts: [handler.getOrCreateTextureLayout(x)],
    outputLayout: handler.createTextureLayoutFromShape(outputShape),
    samplers: ['A'],
    shaderSource,
  };
}

function createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
  const inputTDs = [handler.getOrCreateTextureData(inputs[0], programInfo.inputLayouts[0])];
  return {
    inputTextureDatas: inputTDs,
    outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
    uniformData: {}
  };
}
