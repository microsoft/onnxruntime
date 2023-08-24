// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from '../types';

import {outputVariable, ShaderHelper} from './common';

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length !== 3) {
    throw new Error('Range requires 3 input.');
  }

  if (inputs[0].dims.length !== 0 || inputs[1].dims.length !== 0 || inputs[2].dims.length !== 0) {
    throw new Error('Range requires scalar input.');
  }

  if (inputs[0].dataType !== inputs[1].dataType || inputs[1].dataType !== inputs[2].dataType) {
    throw new Error('Range requires all inputs have the same data type.');
  }
};

const validateInputsContent = (start: number, limit: number, delta: number): void => {
  const sameStartLimit = start === limit;
  const increasingRangeNegativeStep = start < limit && delta < 0;
  const decreasingRangePositiveStep = start > limit && delta > 0;

  if (sameStartLimit || increasingRangeNegativeStep || decreasingRangePositiveStep) {
    throw new Error('Range these inputs\' contents are invalid.');
  }
};

const createRangeProgramInfo =
    (metadata: ProgramMetadata, start: number, limit: number, delta: number, dataType: DataType): ProgramInfo => {
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
        ...metadata,
        getShaderSource,
        outputs: [{dims: outputShape, dataType, gpuDataType: GpuDataType.default}],
        dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
      };
    };

export const range = (context: ComputeContext): void => {
  validateInputs(context.inputs);
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
  validateInputsContent(start, limit, delta);

  const cacheHint = [start, limit, delta].map(x => x.toString()).join('_');
  const metadata: ProgramMetadata = {name: 'Range', inputTypes: [], cacheHint};
  context.compute(
      {...metadata, get: () => createRangeProgramInfo(metadata, start, limit, delta, context.inputs[0].dataType)},
      {inputs: []});
};
