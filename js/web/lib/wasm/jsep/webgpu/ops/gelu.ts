// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {ShaderHelper} from './common';
import {erfImpl} from './unary-op';

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length !== 1) {
    throw new Error('Gelu requires 1 input');
  }
  if (inputs[0].dataType !== DataType.float) {
    throw new Error('Input must be float');
  }
};


const createGeluProgramInfo = (metadata: ProgramMetadata, inputs: readonly TensorView[]): ProgramInfo => {
  const inputShape = inputs[0].dims;
  const outputShape = inputShape.slice(0);
  const outputSize = ShapeUtil.size(outputShape);
  const dataType = 'f32';
  const getShaderSource = (shaderHelper: ShaderHelper) => `
    ${erfImpl('f32')};
    @group(0) @binding(0) var<storage, read> input: array<${dataType}>;
    @group(0) @binding(1) var<storage, read_write> output: array<${dataType}>;
    ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
      var x = input[global_id.x];
      output[global_id.x] = 0.5 * x * (1.0 + erf_vf32(x * 0.7071067811865475));
      }`;
  return {
    ...metadata,
    outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
    getShaderSource,
    dispatchGroup: () => ({x: Math.ceil(outputSize / 64)})
  };
};

const createGeluProgramInfoLoader = (inputs: readonly TensorView[]): ProgramInfoLoader => {
  const metadata: ProgramMetadata = {name: 'Gelu', inputTypes: [GpuDataType.default]};
  return {...metadata, get: () => createGeluProgramInfo(metadata, inputs)};
};

export const gelu = (context: ComputeContext): void => {
  validateInputs(context.inputs);
  // const erfValue = erfImpl(context.inputs[0].getFloat32Array().map(x => x * 0.7071067811865475));
  context.compute(createGeluProgramInfoLoader(context.inputs));
};
