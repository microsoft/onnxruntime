// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// TODO: this is the same naive implementation we use for reduce that has
// performance limitations when the reduced axis is long. Need to add
// a optimized codepath for this.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfoLoader, ProgramMetadata} from '../types';

import {createReduceProgramInfo, ReduceOp} from './reduce';

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length === 0 || inputs.length > 2) {
    throw new Error('ArgMinMaxOp op requires 1 or 2 inputs.');
  }
  if (inputs[0].dataType !== DataType.float) {
    throw new Error('Invalid input type.');
  }
};

export interface ArgMinMaxAttributes extends AttributeWithCacheKey {
  keepDims: boolean;
  axis: number;
  selectLastIndex: number;
}

type ArgMinMaxOp = ReduceOp;

const createArgMinMaxAttributesFromInputs =
    (inputs: readonly TensorView[], attributes: ArgMinMaxAttributes): ArgMinMaxAttributes =>
        createAttributeWithCacheKey(
            {axis: attributes.axis, keepDims: attributes.keepDims, selectLastIndex: attributes.selectLastIndex});

const createReduceProgramInfoLoader =
    (inputs: readonly TensorView[], name: string, attributes: ArgMinMaxAttributes, reduceOp: ArgMinMaxOp):
        ProgramInfoLoader => {
          const updatedAttributes: ArgMinMaxAttributes =
              inputs.length === 1 ? attributes : createArgMinMaxAttributesFromInputs(inputs, attributes);
          const cacheHint = updatedAttributes.cacheKey + inputs.map(x => x.dims.toString()).join('_');
          const metadata: ProgramMetadata = {name, inputTypes: [GpuDataType.default], cacheHint};
          return {
            ...metadata,
            get: () => createReduceProgramInfo(
                metadata, [inputs[0]], reduceOp, [updatedAttributes.axis], DataType.int64, updatedAttributes.keepDims)
          };
        };


export const argMin = (context: ComputeContext, attributes: ArgMinMaxAttributes): void => {
  validateInputs(context.inputs);
  const argMinMaxOp: ArgMinMaxOp = (inputs: TensorView[], axes: number[]): string[] => {
    const idxZero = [];
    for (let k = 0; k < inputs[0].dims.length; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        idxZero.push(`inputIndices[${k}] = 0;`);  // first element
      }
    }
    return [
      `${idxZero.join('\n')}`, 'var value = _A[inputIdx];\nvar bestIndex : i32 = 0;',
      `if (_A[inputIdx] ${
          attributes.selectLastIndex > 0 ? '<=' : '<'} value) {value = _A[inputIdx]; bestIndex = i32(lastIndex);} `,
      '', 'output[global_idx*2] = bestIndex;', 'output[global_idx*2+1] = 0;'
    ];
  };
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ArgMin', attributes, argMinMaxOp), {inputs: [0]});
};

export const argMax = (context: ComputeContext, attributes: ArgMinMaxAttributes): void => {
  validateInputs(context.inputs);
  const argMinMaxOp: ArgMinMaxOp = (inputs: TensorView[], axes: number[]): string[] => {
    const idxZero = [];
    for (let k = 0; k < inputs[0].dims.length; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        idxZero.push(`inputIndices[${k}] = 0;`);  // first element
      }
    }
    return [
      `${idxZero.join('\n')}`, 'var value = _A[inputIdx];\nvar bestIndex : i32 = 0;',
      `if (_A[inputIdx] ${
          attributes.selectLastIndex > 0 ? '>=' : '>'} value) {value = _A[inputIdx]; bestIndex = i32(lastIndex);}`,
      '', 'output[global_idx*2] = bestIndex;', 'output[global_idx*2+1] = 0;'
    ];
  };
  context.compute(createReduceProgramInfoLoader(context.inputs, 'argMax', attributes, argMinMaxOp), {inputs: [0]});
};

export const parseArgMinMaxAttributes = (attributes: Record<string, unknown>): ArgMinMaxAttributes =>
    createAttributeWithCacheKey(attributes as Omit<ArgMinMaxAttributes, keyof AttributeWithCacheKey>);
