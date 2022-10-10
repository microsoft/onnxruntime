// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Graph} from '../../../graph';
import {OperatorAsyncImplementation, OperatorInitialization} from '../../../operators';
import {Tensor} from '../../../tensor';
import {BroadcastUtil, ShapeUtil} from '../../../util';
import {WebGpuInferenceHandler} from '../inference-handler';
import {GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {WORKGROUP_SIZE} from './common';
import {getActicationSnippet, InternalActivationAttributes, parseInternalActivationAttributes} from './fuse-utils';

export const matMul: OperatorAsyncImplementation<InternalActivationAttributes> =
    async(inferenceHandler: WebGpuInferenceHandler, inputs: Tensor[], attributes: InternalActivationAttributes):
        Promise<Tensor[]> => {
          validateInputs(inputs);

          return inferenceHandler.run(createMatmulProgramInfoLoader(inputs, attributes), inputs);
        };

export const parseMatMulAttributes: OperatorInitialization<InternalActivationAttributes> =
    (node: Graph.Node): InternalActivationAttributes => parseInternalActivationAttributes(node.attributes);

const createMatmulProgramMetadata = (hasBias: boolean, cacheHint: string) => ({
  name: 'MatMul',
  inputTypes: hasBias ? [GpuDataType.default, GpuDataType.default, GpuDataType.default] :
                        [GpuDataType.default, GpuDataType.default],
  cacheHint
});

function createMatmulProgramInfo(
    metadata: ProgramMetadata, inputs: Tensor[], activationAttributes: InternalActivationAttributes): ProgramInfo {
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
  const shaderSource = `
  const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;
  const M: u32 = ${M}u;
  const N: u32 = ${N}u;
  const K: u32 = ${K}u;

  @group(0) @binding(0) var<storage, read> a : array<${dataType}>;
  @group(0) @binding(1) var<storage, read> b : array<${dataType}>;
  @group(0) @binding(2) var<storage, read_write> output : array<${dataType}>;

  ${activationFunction}

  @compute @workgroup_size(WORKGROUP_SIZE)
  fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {

    // Guard against out-of-bounds work group sizes
    if (global_id.x >= ${outputSize}u) {
      return;
    }

    let stack = global_id.x / (M * N);
    let mn = global_id.x % (M * N);
    let n = global_id.x % N;
    let m = mn / N;

    let offsetA = stack * (M * K);
    let offsetB = stack * (K * N);

    var value = ${dataType}(0);
    for (var k: u32 = 0u; k<${K}u; k++) {
      value += a[offsetA + m * K + k] * b[offsetB + k * N + n];
    }
    ${applyActivation}
    output[global_id.x] = value;
  }`;
  return {
    ...metadata,
    outputs: [{dims: outputShape, type: inputs[0].type, gpuDataType: GpuDataType.default}],
    shaderSource,
    dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
  };
}

export function createMatmulProgramInfoLoader(
    inputs: Tensor[], activationAttributes: InternalActivationAttributes): ProgramInfoLoader {
  const metadata = createMatmulProgramMetadata(inputs.length > 2, activationAttributes.activationCacheKey);
  return {...metadata, get: () => createMatmulProgramInfo(metadata, inputs, activationAttributes)};
}

const validateInputs = (inputs: Tensor[]): void => {
  if (!inputs || inputs.length !== 2) {
    throw new Error('MatMul requires 2 inputs.');
  }

  if (inputs[0].dims[inputs[0].dims.length - 1] !== inputs[1].dims[inputs[1].dims.length - 2]) {
    throw new Error('shared dimension does not match.');
  }

  if ((inputs[0].type !== 'float32' && inputs[0].type !== 'float64') ||
      (inputs[1].type !== 'float32' && inputs[1].type !== 'float64')) {
    throw new Error('inputs should be float type');
  }

  if (inputs[0].type !== inputs[1].type) {
    throw new Error('inputs types should match');
  }
};
