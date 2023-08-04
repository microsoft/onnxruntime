// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from '../types';

import {ShaderHelper} from './common';

export interface GatherAttributes extends AttributeWithCacheKey {
  axis: number;
}

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length !== 2) {
    throw new Error('Gather requires 2 inputs.');
  }
};

const createGatherProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: GatherAttributes): ProgramInfo => {
      const inputShape = inputs[0].dims;
      const indicesShape = inputs[1].dims;

      const inputRank = inputShape.length;
      const axis = ShapeUtil.normalizeAxis(attributes.axis, inputRank);

      const outputShape = inputShape.slice(0);
      outputShape.splice(axis, 1, ...indicesShape);

      const inputDataType = inputs[0].dataType;
      const block = ShapeUtil.sizeFromDimension(inputShape, axis + 1);
      const elementSize = [DataType.int64, DataType.uint64, DataType.double].includes(inputDataType) ? 2 : 1;
      const indicesElementSize = inputs[1].dataType === DataType.int64 ? 2 : 1;
      const blockSize = elementSize * block;
      const M = ShapeUtil.sizeToDimension(inputShape, axis);
      const N = ShapeUtil.size(indicesShape);
      const dataBatchElements = ShapeUtil.sizeFromDimension(inputShape, axis) * elementSize;
      const gatheredBatchElements = N * block * elementSize;
      const axisDimLimit = inputShape[axis];

      const inputSize = ShapeUtil.size(inputShape) * elementSize;
      const outputSize = ShapeUtil.size(outputShape) * elementSize;

      const totalGathers = M * N;
      // int64 indices would be treated as little endian i32 with assumption they fall in i32 limits
      // That assumption is safe as it's not possible to allocate >2gb buffer for input tensor
      // Input data will be treated as u32 or two u32 for 8-byte tensors
      const getShaderSource = (shaderHelper: ShaderHelper) => `
  const N: u32 = ${N};
  const elementSize: u32 = ${elementSize};
  const indicesElementSize: u32 = ${indicesElementSize};

  @group(0) @binding(0) var<storage, read> input : array<u32>;
  @group(0) @binding(1) var<storage, read> inputIndices : array<i32>;
  @group(0) @binding(2) var<storage, read_write> output: array<u32>;

  ${shaderHelper.mainStart()}
    let batch: u32 = global_idx / N;
    let i: u32 = global_idx % N;

    let srcOffsetBatch: u32 = batch * ${dataBatchElements};
    let dstOffsetBatch: u32 = batch * ${gatheredBatchElements};
    var idx = inputIndices[i * indicesElementSize];
    if (idx < 0) {
        idx = idx + ${axisDimLimit};
    }

    let srcOffset = srcOffsetBatch + u32(idx) * ${blockSize};
    let dstOffset = dstOffsetBatch + i * ${blockSize};
    if (srcOffset >= ${inputSize}) {
        return;
    }
    if (dstOffset >= ${outputSize}) {
        return;
    }
    for (var j: u32 = 0; j < ${blockSize}; j++) {
        output[dstOffset + j] = input[srcOffset + j];
    }
  }`;
      return {
        ...metadata,
        outputs: [
          {dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default},
        ],
        getShaderSource,
        dispatchGroup: () => ({x: Math.ceil(totalGathers / 64 /* workgroup size */)})
      };
    };

export const parseGatherAttributes = (attributes: Record<string, unknown>): GatherAttributes =>
    createAttributeWithCacheKey({axis: attributes.axis as number});

export const gather = (context: ComputeContext, attributes: GatherAttributes): void => {
  const inputs = context.inputs;
  validateInputs(inputs);

  const metadata = {
    name: 'Gather',
    inputTypes: [GpuDataType.default, GpuDataType.default],
    cacheHint: attributes.cacheKey + inputs[0].dataType.toString(10) + inputs[1].dataType.toString(10),
  };

  context.compute(createGatherProgramInfo(metadata, context.inputs, attributes));
};
