// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'assert/strict';
import {Tensor} from 'onnxruntime-common';

/**
 * A list of numerical types that are compatible with JavaScript 'number' value.
 *
 * 3 elements in each list are:
 *  - type: a string representing the type name,
 *  - typedArrayConstructor: the built-in typed array constructor for the type,
 *  - canBeInferredFromType: whether the type can be inferred from the type name.
 */
export const NUMBER_COMPATIBLE_NUMERICAL_TYPES = [
  ['int8', Int8Array, true] as const,
  ['uint8', Uint8Array, true] as const,
  ['int16', Int16Array, true] as const,
  ['uint16', Uint16Array, true] as const,
  ['int32', Int32Array, true] as const,
  ['uint32', Uint32Array, true] as const,
  ['float32', Float32Array, true] as const,
  ['float64', Float64Array, true] as const,
];

/**
 * Big integer types
 */
export const BIGINT_TYPES = [
  ['int64', BigInt64Array, true] as const,
  ['uint64', BigUint64Array, true] as const,
];

/**
 * float16 type, data represented by Uint16Array
 */
export const FLOAT16_TYPE = ['float16', Uint16Array, false] as const;

/**
 * A list of all numerical types.
 *
 * not including string and bool.
 */
export const ALL_NUMERICAL_TYPES = [...NUMBER_COMPATIBLE_NUMERICAL_TYPES, ...BIGINT_TYPES, FLOAT16_TYPE];

/**
 * a helper function to assert that a value is an array of a certain type
 */
export const assertIsArrayOf = (value: unknown, type: 'string'|'number'|'boolean'): void => {
  assert(Array.isArray(value), 'array should be an array');
  for (let i = 0; i < value.length; i++) {
    assert.equal(typeof value[i], type, `array should be an array of ${type}s`);
  }
};

/**
 * the 'TensorAny' is a type allows skip typescript type checking for Tensor.
 *
 * This allows to write test code to pass invalid parameters to Tensor constructor and check the behavior.
 */
export const TensorAny = Tensor as unknown as {new (...args: unknown[]): Tensor};
