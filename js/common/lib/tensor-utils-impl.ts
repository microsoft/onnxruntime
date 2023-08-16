// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from './tensor.js';

/**
 * calculate size from dims.
 *
 * @param dims the dims array. May be an illegal input.
 */
export const calculateSize = (dims: readonly unknown[]): number => {
  let size = 1;
  for (let i = 0; i < dims.length; i++) {
    const dim = dims[i];
    if (typeof dim !== 'number' || !Number.isSafeInteger(dim)) {
      throw new TypeError(`dims[${i}] must be an integer, got: ${dim}`);
    }
    if (dim < 0) {
      throw new RangeError(`dims[${i}] must be a non-negative integer, got: ${dim}`);
    }
    size *= dim;
  }
  return size;
};

/**
 * implementation of Tensor.reshape()
 */
export const tensorReshape = (tensor: Tensor, dims: readonly number[]): Tensor =>
    new Tensor(tensor.type, tensor.data, dims);
