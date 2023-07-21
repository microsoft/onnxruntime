// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {BroadcastUtil, ShapeUtil} from '../../util';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {ShaderHelper} from './common';
import {getActicationSnippet, InternalActivationAttributes} from './fuse-utils';


const createMatmulProgramMetadata = (hasBias: boolean, cacheHint: string) => ({
  name: 'MatMul',
  inputTypes: hasBias ? [GpuDataType.default, GpuDataType.default, GpuDataType.default] :
                        [GpuDataType.default, GpuDataType.default],
  cacheHint
});

const createMatmulProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], activationAttributes: InternalActivationAttributes):
        ProgramInfo => {
          const aShape = inputs[0].dims;
          const bShape = inputs[1].dims;
          const outputShape = BroadcastUtil.calcShape(aShape, bShape, true);
          if (!outputShape) {
            throw new Error('Can\'t use matmul on the given tensors');
          }
          const outputSize = ShapeUtil.size(outputShape);
          // TODO: support broadcasting

          const dataType = 'f32';  // TODO: support other data type
          const {activationFunction, applyActivation} = getActicationSnippet(activationAttributes);

          const M = outputShape[outputShape.length - 2];
          const K = aShape[aShape.length - 1];
          const N = outputShape[outputShape.length - 1];
          const getShaderSource = (shaderHelper: ShaderHelper) => `
  const M: u32 = ${M}u;
  const N: u32 = ${N}u;
  const K: u32 = ${K}u;

  @group(0) @binding(0) var<storage, read> a : array<${dataType}>;
  @group(0) @binding(1) var<storage, read> b : array<${dataType}>;
  @group(0) @binding(2) var<storage, read_write> output : array<${dataType}>;

  ${activationFunction}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

    let stack = global_idx / (M * N);
    let mn = global_idx % (M * N);
    let n = global_idx % N;
    let m = mn / N;

    let offsetA = stack * (M * K);
    let offsetB = stack * (K * N);

    var value = ${dataType}(0);
    for (var k: u32 = 0u; k<${K}u; k++) {
      value += a[offsetA + m * K + k] * b[offsetB + k * N + n];
    }
    ${applyActivation}
    output[global_idx] = value;
  }`;
          return {
            ...metadata,
            outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
            getShaderSource,
            dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
          };
        };

export const createMatmulProgramInfoLoader =
    (inputs: readonly TensorView[], activationAttributes: InternalActivationAttributes): ProgramInfoLoader => {
      const metadata = createMatmulProgramMetadata(inputs.length > 2, activationAttributes.activationCacheKey);
      return {...metadata, get: () => createMatmulProgramInfo(metadata, inputs, activationAttributes)};
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

  context.compute(createMatmulProgramInfoLoader(context.inputs, {activation: '', activationCacheKey: ''}));
};
