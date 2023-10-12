// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {env} from 'onnxruntime-common';

import {DataType} from '../../../wasm-common';
import {ComputeContext, ProgramInfo} from '../types';

import {outputVariable, ShaderHelper} from './common';

const validateInputsContent = (start: number, limit: number, delta: number): void => {
  const sameStartLimit = start === limit;
  const increasingRangeNegativeStep = start < limit && delta < 0;
  const decreasingRangePositiveStep = start > limit && delta > 0;

  if (sameStartLimit || increasingRangeNegativeStep || decreasingRangePositiveStep) {
    throw new Error('Range these inputs\' contents are invalid.');
  }
};

const createRangeProgramInfo = (start: number, limit: number, delta: number, dataType: DataType): ProgramInfo => {
  const numElements = Math.abs(Math.ceil((limit - start) / delta));
  const outputShape: number[] = [numElements];
  const outputSize = numElements;

  const output = outputVariable('output', dataType, outputShape);
  const wgslType = output.type.storage;

  const getShaderSource = (shaderHelper: ShaderHelper) => `
        ${shaderHelper.declareVariables(output)}
        ${shaderHelper.mainStart()}
        ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
        output[global_idx] = ${wgslType}(${start}) + ${wgslType}(global_idx) * ${wgslType}(${delta});
      }`;
  return {
    name: 'Range',
    shaderCache: {hint: [start, limit, delta].map(x => x.toString()).join('_')},
    getShaderSource,
    getRunData: () => (
        {outputs: [{dims: outputShape, dataType}],
         dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)}})
  };
};

export const range = (context: ComputeContext): void => {
  let start = 0;
  let limit = 0;
  let delta = 0;
  if (context.inputs[0].dataType === DataType.int32) {
    start = context.inputs[0].getInt32Array()[0];
    limit = context.inputs[1].getInt32Array()[0];
    delta = context.inputs[2].getInt32Array()[0];
  } else if (context.inputs[0].dataType === DataType.float) {
    start = context.inputs[0].getFloat32Array()[0];
    limit = context.inputs[1].getFloat32Array()[0];
    delta = context.inputs[2].getFloat32Array()[0];
  }
  if (env.webgpu.validateInputContent) {
    validateInputsContent(start, limit, delta);
  }

  context.compute(createRangeProgramInfo(start, limit, delta, context.inputs[0].dataType), {inputs: []});
};
