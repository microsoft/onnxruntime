// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor as TensorInterface} from './tensor';

type SupportedTypedArrayConstructors = Float32ArrayConstructor|Uint8ArrayConstructor|Int8ArrayConstructor|
    Uint16ArrayConstructor|Int16ArrayConstructor|Int32ArrayConstructor|BigInt64ArrayConstructor|Uint8ArrayConstructor|
    Float64ArrayConstructor|Uint32ArrayConstructor|BigUint64ArrayConstructor;
type SupportedTypedArray = InstanceType<SupportedTypedArrayConstructors>;

// a runtime map that maps type string to TypedArray constructor. Should match Tensor.DataTypeMap.
const NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP = new Map<string, SupportedTypedArrayConstructors>([
  ['float32', Float32Array],
  ['uint8', Uint8Array],
  ['int8', Int8Array],
  ['uint16', Uint16Array],
  ['int16', Int16Array],
  ['int32', Int32Array],
  ['int64', BigInt64Array],
  ['bool', Uint8Array],
  ['float64', Float64Array],
  ['uint32', Uint32Array],
  ['uint64', BigUint64Array],
]);

// a runtime map that maps type string to TypedArray constructor. Should match Tensor.DataTypeMap.
const NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP = new Map<Function, TensorInterface.Type>([
  [Float32Array, 'float32'],
  [Uint8Array, 'uint8'],
  [Int8Array, 'int8'],
  [Uint16Array, 'uint16'],
  [Int16Array, 'int16'],
  [Int32Array, 'int32'],
  [BigInt64Array, 'int64'],
  [Float64Array, 'float64'],
  [Uint32Array, 'uint32'],
  [BigUint64Array, 'uint64'],
]);


/**
 * calculate size from dims.
 *
 * @param dims the dims array. May be an illegal input.
 */
const calculateSize = (dims: readonly unknown[]): number => {
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

export class Tensor implements TensorInterface {
  //#region constructors
  constructor(
      type: TensorInterface.Type, data: TensorInterface.DataType|readonly number[]|readonly boolean[],
      dims?: readonly number[]);
  constructor(data: TensorInterface.DataType|readonly boolean[], dims?: readonly number[]);
  constructor(
      arg0: TensorInterface.Type|TensorInterface.DataType|readonly boolean[],
      arg1?: TensorInterface.DataType|readonly number[]|readonly boolean[], arg2?: readonly number[]) {
    let type: TensorInterface.Type;
    let data: TensorInterface.DataType;
    let dims: typeof arg1|typeof arg2;
    // check whether arg0 is type or data
    if (typeof arg0 === 'string') {
      //
      // Override: constructor(type, data, ...)
      //
      type = arg0;
      dims = arg2;
      if (arg0 === 'string') {
        // string tensor
        if (!Array.isArray(arg1)) {
          throw new TypeError('A string tensor\'s data must be a string array.');
        }
        // we don't check whether every element in the array is string; this is too slow. we assume it's correct and
        // error will be populated at inference
        data = arg1;
      } else {
        // numeric tensor
        const typedArrayConstructor = NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP.get(arg0);
        if (typedArrayConstructor === undefined) {
          throw new TypeError(`Unknown tensor type: ${arg0}.`);
        }
        if (Array.isArray(arg1)) {
          // use 'as any' here because TypeScript's check on type of 'SupportedTypedArrayConstructors.from()' produces
          // incorrect results.
          // 'typedArrayConstructor' should be one of the typed array prototype objects.
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          data = (typedArrayConstructor as any).from(arg1);
        } else if (arg1 instanceof typedArrayConstructor) {
          data = arg1;
        } else {
          throw new TypeError(`A ${type} tensor's data must be type of ${typedArrayConstructor}`);
        }
      }
    } else {
      //
      // Override: constructor(data, ...)
      //
      dims = arg1;
      if (Array.isArray(arg0)) {
        // only boolean[] and string[] is supported
        if (arg0.length === 0) {
          throw new TypeError('Tensor type cannot be inferred from an empty array.');
        }
        const firstElementType = typeof arg0[0];
        if (firstElementType === 'string') {
          type = 'string';
          data = arg0;
        } else if (firstElementType === 'boolean') {
          type = 'bool';
          // 'arg0' is of type 'boolean[]'. Uint8Array.from(boolean[]) actually works, but typescript thinks this is
          // wrong type. We use 'as any' to make it happy.
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          data = Uint8Array.from(arg0 as any[]);
        } else {
          throw new TypeError(`Invalid element type of data array: ${firstElementType}.`);
        }
      } else {
        // get tensor type from TypedArray
        const mappedType = NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP.get(arg0.constructor);
        if (mappedType === undefined) {
          throw new TypeError(`Unsupported type for tensor data: ${arg0.constructor}.`);
        }
        type = mappedType;
        data = arg0 as SupportedTypedArray;
      }
    }

    // type and data is processed, now processing dims
    if (dims === undefined) {
      // assume 1-D tensor if dims omitted
      dims = [data.length];
    } else if (!Array.isArray(dims)) {
      throw new TypeError('A tensor\'s dims must be a number array');
    }

    // perform check
    const size = calculateSize(dims);
    if (size !== data.length) {
      throw new Error(`Tensor's size(${size}) does not match data length(${data.length}).`);
    }

    this.dims = dims as readonly number[];
    this.type = type;
    this.data = data;
    this.size = size;
  }
  //#endregion

  //#region fields
  readonly dims: readonly number[];
  readonly type: TensorInterface.Type;
  readonly data: TensorInterface.DataType;
  readonly size: number;
  //#endregion

  //#region tensor utilities
  reshape(dims: readonly number[]): TensorInterface {
    return new Tensor(this.type, this.data, dims);
  }
  //#endregion
}
