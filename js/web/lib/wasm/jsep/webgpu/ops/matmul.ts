// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {BroadcastUtil} from '../../util';
import {ComputeContext, GpuDataType, ProgramInfoLoader} from '../types';

import {createMatmulProgramInfo} from './3rd-party/matmul_packed_webgpu';
import {InternalActivationAttributes} from './fuse-utils';


const createMatmulProgramMetadata = (hasBias: boolean, cacheHint: string) => ({
  name: 'MatMul',
  inputTypes: hasBias ? [GpuDataType.default, GpuDataType.default, GpuDataType.default] :
                        [GpuDataType.default, GpuDataType.default],
  cacheHint
});

export const createMatmulProgramInfoLoader =
    (inputs: readonly TensorView[], activationAttributes: InternalActivationAttributes, outputShape: readonly number[],
     reshapedOutputShape?: readonly number[]): ProgramInfoLoader => {
      const metadata = createMatmulProgramMetadata(inputs.length > 2, activationAttributes.activationCacheKey);
      return {
        ...metadata,
        get: () => createMatmulProgramInfo(metadata, inputs, activationAttributes, outputShape, reshapedOutputShape)
      };
    };

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length !== 2) {
    throw new Error('MatMul requires 2 inputs.');
  }

  if (inputs[0].dims[inputs[0].dims.length - 1] !== inputs[1].dims[inputs[1].dims.length - 2]) {
    throw new Error('shared dimension does not match.');
  }

  if (inputs[0].dataType !== DataType.float || inputs[1].dataType !== DataType.float) {
    throw new Error('inputs should be float type');
  }
};

export const matMul = (context: ComputeContext): void => {
  validateInputs(context.inputs);
  const outputShape = BroadcastUtil.calcShape(context.inputs[0].dims, context.inputs[1].dims, true);
  if (!outputShape) {
    throw new Error('Can\'t use matmul on the given tensors');
  }
  context.compute(createMatmulProgramInfoLoader(context.inputs, {activation: '', activationCacheKey: ''}, outputShape));
};
