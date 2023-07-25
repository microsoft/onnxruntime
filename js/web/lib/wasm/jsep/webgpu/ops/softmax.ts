// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// TODO: this is the same naive implementation we use for reduce that has
// performance limitations when the reduced axis is long. Need to add
// a optimized codepath for this.

import { DataType } from '../../../wasm-common';
import { ShapeUtil } from '../../util';
import { TensorView } from '../../tensor';
import { AttributeWithCacheKey, createAttributeWithCacheKey } from '../attribute-with-cache-key';
import { ComputeContext, GpuDataType, ProgramInfo } from '../types';
import { ShaderHelper } from './common';

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length !== 1) {
    throw new Error('Softmax op requires 1 input.');
  }
  if (inputs[0].dataType !== DataType.float) {
    throw new Error('Softmax input needs to be float.');
  }
};

export interface SoftmaxAttributes extends AttributeWithCacheKey {
  readonly axis: number;
}

export const SoftmaxProgramMetadata = {
  name: 'Softmax',
  inputTypes: [GpuDataType.default]
};


const createSoftmaxProgramInfo = (input: TensorView, attributes: SoftmaxAttributes): ProgramInfo => {
  const dataType = 'f32';
  const shape = input.dims;
  const outputSize = ShapeUtil.size(shape);
  const WG = 64;
  var axis = attributes.axis;
  if (axis < 0) {
    axis = shape.length + axis;
  }
  if (axis < shape.length - 1) {
    throw new Error('softmax only supports last axis for now.');
  }

  const cols = shape[axis];
  const rows = outputSize / cols;

  const getShaderSource = (shaderHelper: ShaderHelper) => `
      var<workgroup> rowMaxShared : ${dataType};
      var<workgroup> rowSumShared : ${dataType};
      var<workgroup> threadShared : array<${dataType}, ${WG}>;

      @group(0) @binding(0) var<storage, read> x : array<${dataType}>;
      @group(0) @binding(1) var<storage, read_write> result : array<${dataType}>;

      fn getValue(row: i32, col: i32, row_stride: i32) -> ${dataType} {
        let index = row * row_stride + col;
        return x[index];
      }

      fn setValue(row: i32, col: i32, row_stride: i32, value: ${dataType}) {
        let index = row * row_stride + col;
        result[index] = value;
      }

      @compute @workgroup_size(${WG}, 1, 1)
      fn main(@builtin(local_invocation_id) local_id : vec3<u32>, @builtin(global_invocation_id) global_id : vec3u) {
        let gindex = i32(global_id.x);
        let lindex = i32(local_id.x);
        const wg = ${WG};
        let row = gindex / wg;
        let cols = ${cols};
        let row_stride : i32 = ${cols};

        // find the rows max
        var threadMax = -3.402823e+38f; // 6.2.4 in wgsl spec
        for (var col = lindex; col < cols; col += wg) {
          let value = getValue(row, col, row_stride);
          threadMax = max(threadMax, value);
        }
        if (lindex < cols) {
          threadShared[lindex] = threadMax;
        }
        workgroupBarrier();

        var reduceSize = min(cols, wg);
        for (var currSize = reduceSize >> 1;  currSize > 0; currSize = reduceSize >> 1) {
          reduceSize = currSize + (reduceSize & 1);
          if (lindex < currSize) {
            threadShared[lindex] = max(threadShared[lindex], threadShared[lindex + reduceSize]);
          }
          workgroupBarrier();
        }
        if (lindex == 0) {
          rowMaxShared = threadShared[0];
        }
        workgroupBarrier();

        // find the rows sum
        var threadSum = 0.0;
        for (var col = lindex; col < cols; col += wg) {
          let subExp = exp(getValue(row, col, row_stride) - rowMaxShared);
          threadSum += subExp;
        }
        threadShared[lindex] = threadSum;
        workgroupBarrier();

        for (var currSize = wg >> 1;  currSize > 0; currSize = currSize >> 1) {
          if (lindex < currSize) {
            threadShared[lindex] = threadShared[lindex] + threadShared[lindex + currSize];
          }
          workgroupBarrier();
        }
        if (lindex == 0) {
          rowSumShared = threadShared[0];
        }
        workgroupBarrier();

        // calculate final value for each element in the row
        for (var col = lindex; col < cols; col += wg) {
          let value = exp(getValue(row, col, row_stride) - rowMaxShared) / rowSumShared;
          setValue(row, col, row_stride, value);
        }
      }`;
  return {
    ...SoftmaxProgramMetadata,
    outputs: [{ dims: shape, dataType: input.dataType, gpuDataType: GpuDataType.default }],
    getShaderSource,
    dispatchGroup: () => ({ x: rows })
  };
};


export const softmax = (context: ComputeContext, attributes: SoftmaxAttributes): void => {
  validateInputs(context.inputs);
  context.compute({
    ...SoftmaxProgramMetadata,
    cacheHint: attributes.cacheKey,
    get: () => createSoftmaxProgramInfo(context.inputs[0], attributes)
  });
};

export const parseSoftmaxAttributes = (attributes: Record<string, unknown>): SoftmaxAttributes =>
  createAttributeWithCacheKey({ axis: attributes.axis as number });
