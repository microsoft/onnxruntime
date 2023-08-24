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

const getShaderSourceForComplexBroadcast =
    (aShape: readonly number[], bShape: readonly number[], outputShape: readonly number[],
     activationAttributes: InternalActivationAttributes) => {
      const maxDims = outputShape.length;
      const outputSize = ShapeUtil.size(outputShape);
      const {activationFunction, applyActivation} = getActicationSnippet(activationAttributes);
      const dataType = 'f32';
      const K = aShape[aShape.length - 1];

      const aShapePadded = [...Array(maxDims - aShape.length).fill(1), ...aShape];
      const bShapePadded = [...Array(maxDims - bShape.length).fill(1), ...bShape];

      return (shaderHelper: ShaderHelper) => `
  const MAX_DIMS : u32 = ${maxDims};
  const aShape = array<u32, ${maxDims}>(${aShapePadded.join(',')});
  const bShape = array<u32, ${maxDims}>(${bShapePadded.join(',')});
  const outShape = array<u32, ${maxDims}>(${outputShape.join(',')});

fn coordToIndex(coord: array<u32, ${maxDims}>, shape: array<u32, ${maxDims}>) -> u32 {
    var index: u32 = 0;
    var stride: u32 = 1;
    for (var i: i32 = i32(MAX_DIMS) - 1; i >= 0; i = i - 1) {
        index = index + (coord[i] % shape[i]) * stride;
        stride = stride * shape[i];
    }
    return index;
}

fn mapIndices(outIndex: u32, k: u32, outShape: array<u32, ${maxDims}>, shapeA: array<u32, ${
                 maxDims}>, shapeB: array<u32, ${maxDims}>) -> vec2<u32> {
    var coord = array<u32, ${maxDims}>(${outputShape.map(_ => 0).join(',')});

    var index = outIndex;
    // we can't use u32 in backwards for loop to 0
    for (var i: i32 = i32(MAX_DIMS) - 1; i >= 0; i = i - 1) {
        coord[i] = index % outShape[i];
        index = index / outShape[i];
    }

    // Map the output coordinate to coordinates in A and B
    var aCoord: array<u32, ${maxDims}> = coord;
    aCoord[MAX_DIMS - 1] = k;

    var bCoord: array<u32, ${maxDims}> = coord;
    bCoord[MAX_DIMS - 2] = k;

    let indA: u32 = coordToIndex(aCoord, shapeA);
    let indB: u32 = coordToIndex(bCoord, shapeB);

    return vec2<u32>(indA, indB);
}
  const K: u32 = ${K}u;

  @group(0) @binding(0) var<storage, read> a : array<${dataType}>;
  @group(0) @binding(1) var<storage, read> b : array<${dataType}>;
  @group(0) @binding(2) var<storage, read_write> output : array<${dataType}>;

  ${activationFunction}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

    var value = ${dataType}(0);
    for (var k: u32 = 0u; k<${K}u; k++) {
      let indices = mapIndices(global_idx, k, outShape, aShape, bShape);
      value += a[indices[0]] * b[indices[1]];
    }
    ${applyActivation}
    output[global_idx] = value;
  }`;
    };

export const getSimpleBroadcastAForMatMul = (aShape: readonly number[], bShape: readonly number[]): string|null => {
  const rankA = aShape.length;
  const rankB = bShape.length;

  if (rankA <= 2) {
    return 'm * K';
  }

  if (rankA === 3 && rankB === 4) {
    return `m * K + stack % ${aShape[0]} * (M * K);`;
  }

  if (rankA === 4 && rankB === 5) {
    return `m * K + (stack) % ${aShape[0] * aShape[1]} * (M * K);`;
  }

  return null;
};

export const createMatmulProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], activationAttributes: InternalActivationAttributes):
        ProgramInfo => {
          const aShape = inputs[0].dims;
          const bShape = inputs[1].dims;
          const outputShape = BroadcastUtil.calcShape(aShape, bShape, true);
          if (!outputShape) {
            throw new Error('Can\'t use matmul on the given tensors');
          }
          const outputSize = ShapeUtil.size(outputShape);
          const broadcastA = (aShape.length < outputShape.length || aShape[aShape.length - 2] === 1);
          const broadcastB = (bShape.length < outputShape.length || bShape[bShape.length - 1] === 1);

          const simpleBroadcastA = getSimpleBroadcastAForMatMul(aShape, bShape);
          let getShaderSource;
          // let's see if we can use simple broadcasting for given shapes
          if ((broadcastA && !simpleBroadcastA) || (broadcastB && bShape.length > 2)) {
            getShaderSource = getShaderSourceForComplexBroadcast(
                aShape,
                bShape,
                outputShape,
                activationAttributes,
            );
          } else {
            const dataType = 'f32';  // TODO: support other data type
            const M = outputShape[outputShape.length - 2];
            const K = aShape[aShape.length - 1];
            const N = outputShape[outputShape.length - 1];
            const {activationFunction, applyActivation} = getActicationSnippet(activationAttributes);
            getShaderSource = (shaderHelper: ShaderHelper) => `
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

    let offsetA = ${broadcastA ? simpleBroadcastA : 'stack * (M * K) + m * K'};
    let offsetB = ${broadcastB ? 'n' : 'stack * (K * N) + n'};

    var value = ${dataType}(0);
    for (var k: u32 = 0u; k<${K}u; k++) {
      value += a[offsetA + k] * b[offsetB + k * N];
    }
    ${applyActivation}
    output[global_idx] = value;
  }`;
          }
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
