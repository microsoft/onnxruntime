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

const createArgMinMaxAttributesFromInputs =
    (inputs: readonly TensorView[], attributes: ArgMinMaxAttributes): ArgMinMaxAttributes =>
        createAttributeWithCacheKey(
            {axis: attributes.axis, keepDims: attributes.keepDims, selectLastIndex: attributes.selectLastIndex});

const createArgMinMaxProgramInfoLoader =
    (inputs: readonly TensorView[], name: string, attributes: ArgMinMaxAttributes, reduceOp: ReduceOp):
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
  const argMinMaxOp: ReduceOp = (input, output, axes) => {
    const idxZero = [];
    for (let k = 0; k < input.shape.length; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        idxZero.push(`inputIndices[${k}] = 0;`);  // first element
      }
    }
    return [
      `${idxZero.join('\n')}`, `var value = ${input.getByOffset('inputOffset')};\nvar bestIndex : i32 = 0;`,
      `if (${input.getByOffset('inputOffset')} ${attributes.selectLastIndex > 0 ? '<=' : '<'} value) {
         value = ${input.getByOffset('inputOffset')};
         bestIndex = i32(lastIndex);
       }`,
      '', output.setByOffset('global_idx', 'bestIndex')
    ];
  };
  context.compute(createArgMinMaxProgramInfoLoader(context.inputs, 'ArgMin', attributes, argMinMaxOp), {inputs: [0]});
};

export const argMax = (context: ComputeContext, attributes: ArgMinMaxAttributes): void => {
  validateInputs(context.inputs);
  const argMinMaxOp: ReduceOp = (input, output, axes) => {
    const idxZero = [];
    for (let k = 0; k < input.shape.length; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        idxZero.push(`inputIndices[${k}] = 0;`);  // first element
      }
    }
    return [
      `${idxZero.join('\n')}`, `var value = ${input.getByOffset('inputOffset')};\nvar bestIndex : i32 = 0;`,
      `if (${input.getByOffset('inputOffset')} ${attributes.selectLastIndex > 0 ? '>=' : '>'} value) {
         value = ${input.getByOffset('inputOffset')};
         bestIndex = i32(lastIndex);
       }`,
      '', output.setByOffset('global_idx', 'bestIndex')
    ];
  };
  context.compute(createArgMinMaxProgramInfoLoader(context.inputs, 'argMax', attributes, argMinMaxOp), {inputs: [0]});
};

export const parseArgMinMaxAttributes = (attributes: Record<string, unknown>): ArgMinMaxAttributes =>
    createAttributeWithCacheKey(attributes as Omit<ArgMinMaxAttributes, keyof AttributeWithCacheKey>);
