// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TypedTensorUtils} from './tensor-utils';

/**
 * represent a basic tensor with specified dimensions and data type.
 */
interface TypedTensorBase<T extends Tensor.Type> {
  /**
   * Get the dimensions of the tensor.
   */
  readonly dims: readonly number[];
  /**
   * Get the data type of the tensor.
   */
  readonly type: T;
  /**
   * Get the buffer data of the tensor.
   */
  readonly data: Tensor.DataTypeMap[T];
}

export declare namespace Tensor {
  interface DataTypeMap {
    float32: Float32Array;
    uint8: Uint8Array;
    int8: Int8Array;
    uint16: Uint16Array;
    int16: Int16Array;
    int32: Int32Array;
    int64: BigInt64Array;
    string: string[];
    bool: Uint8Array;
    float16: never;  // hold on using Uint16Array before we have a concrete solution for float 16
    float64: Float64Array;
    uint32: Uint32Array;
    uint64: BigUint64Array;
    // complex64: never;
    // complex128: never;
    // bfloat16: never;
  }

  interface ElementTypeMap {
    float32: number;
    uint8: number;
    int8: number;
    uint16: number;
    int16: number;
    int32: number;
    int64: number;  // may lose precision
    string: string;
    bool: boolean;
    float16: never;  // hold on before we have a concret solution for float 16
    float64: number;
    uint32: number;
    uint64: number;  // may lose precision
    // complex64: never;
    // complex128: never;
    // bfloat16: never;
  }

  type DataType = DataTypeMap[Type];
  type ElementType = ElementTypeMap[Type];

  /**
   * represent the data type of a tensor
   */
  export type Type = keyof DataTypeMap;
}

export interface TypedTensor<T extends Tensor.Type> extends TypedTensorBase<T>, TypedTensorUtils<T> {}
export interface Tensor extends TypedTensorBase<Tensor.Type>, TypedTensorUtils<Tensor.Type> {}

export interface TensorConstructor {
  //#region specify element type
  /**
   * Construct a new string tensor object from the given type, data and dims.
   *
   * @type Specify the element type.
   * @data Specify the tensor data
   * @dims Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(type: 'string', data: Tensor.DataTypeMap['string']|readonly string[],
      dims?: readonly number[]): TypedTensor<'string'>;

  /**
   * Construct a new bool tensor object from the given type, data and dims.
   *
   * @type Specify the element type.
   * @data Specify the tensor data
   * @dims Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(type: 'bool', data: Tensor.DataTypeMap['bool']|readonly boolean[], dims?: readonly number[]): TypedTensor<'bool'>;

  /**
   * Construct a new numeric tensor object from the given type, data and dims.
   *
   * @type Specify the element type.
   * @data Specify the tensor data
   * @dims Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new<T extends Exclude<Tensor.Type, 'string'|'bool'>>(
      type: T, data: Tensor.DataTypeMap[T]|readonly number[], dims?: readonly number[]): TypedTensor<T>;
  //#endregion

  //#region infer element types

  /**
   * Construct a new float32 tensor object from the given data and dims.
   *
   * @data Specify the tensor data
   * @dims Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: Float32Array, dims?: readonly number[]): TypedTensor<'float32'>;

  /**
   * Construct a new int8 tensor object from the given data and dims.
   *
   * @data Specify the tensor data
   * @dims Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: Int8Array, dims?: readonly number[]): TypedTensor<'int8'>;

  /**
   * Construct a new uint8 tensor object from the given data and dims.
   *
   * @data Specify the tensor data
   * @dims Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: Uint8Array, dims?: readonly number[]): TypedTensor<'uint8'>;

  /**
   * Construct a new uint16 tensor object from the given data and dims.
   *
   * @data Specify the tensor data
   * @dims Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: Uint16Array, dims?: readonly number[]): TypedTensor<'uint16'>;

  /**
   * Construct a new int16 tensor object from the given data and dims.
   *
   * @data Specify the tensor data
   * @dims Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: Int16Array, dims?: readonly number[]): TypedTensor<'int16'>;

  /**
   * Construct a new int32 tensor object from the given data and dims.
   *
   * @data Specify the tensor data
   * @dims Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: Int32Array, dims?: readonly number[]): TypedTensor<'int32'>;

  /**
   * Construct a new int64 tensor object from the given data and dims.
   *
   * @data Specify the tensor data
   * @dims Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: BigInt64Array, dims?: readonly number[]): TypedTensor<'int64'>;

  /**
   * Construct a new string tensor object from the given data and dims.
   *
   * @data Specify the tensor data
   * @dims Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: readonly string[], dims?: readonly number[]): TypedTensor<'string'>;

  /**
   * Construct a new bool tensor object from the given data and dims.
   *
   * @data Specify the tensor data
   * @dims Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: readonly boolean[], dims?: readonly number[]): TypedTensor<'bool'>;

  /**
   * Construct a new float64 tensor object from the given data and dims.
   *
   * @data Specify the tensor data
   * @dims Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: Float64Array, dims?: readonly number[]): TypedTensor<'float64'>;

  /**
   * Construct a new uint32 tensor object from the given data and dims.
   *
   * @data Specify the tensor data
   * @dims Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: Uint32Array, dims?: readonly number[]): TypedTensor<'uint32'>;

  /**
   * Construct a new uint64 tensor object from the given data and dims.
   *
   * @data Specify the tensor data
   * @dims Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: BigUint64Array, dims?: readonly number[]): TypedTensor<'uint64'>;

  //#endregion

  //#region fall back to non-generic tensor type declaration

  /**
   * Construct a new tensor object from the given type, data and dims.
   *
   * @type Specify the element type.
   * @data Specify the tensor data
   * @dims Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(type: Tensor.Type, data: Tensor.DataType|readonly number[]|readonly boolean[], dims?: readonly number[]): Tensor;

  /**
   * Construct a new tensor object from the given data and dims.
   *
   * @data Specify the tensor data
   * @dims Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: Tensor.DataType, dims?: readonly number[]): Tensor;
  //#endregion
}

// eslint-disable-next-line @typescript-eslint/no-redeclare

type SupportedTypedArrayConstructors = Float32ArrayConstructor|Uint8ArrayConstructor|Int8ArrayConstructor|
    Uint16ArrayConstructor|Int16ArrayConstructor|Int32ArrayConstructor|BigInt64ArrayConstructor|Uint8ArrayConstructor|
    Float64ArrayConstructor|Uint32ArrayConstructor|BigUint64ArrayConstructor;
type SupportedTypedArray = InstanceType<SupportedTypedArrayConstructors>;

const isBigInt64ArrayAvailable = typeof BigInt64Array !== 'undefined' && typeof BigInt64Array.from === 'function';
const isBigUint64ArrayAvailable = typeof BigUint64Array !== 'undefined' && typeof BigUint64Array.from === 'function';

// a runtime map that maps type string to TypedArray constructor. Should match Tensor.DataTypeMap.
const NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP = new Map<string, SupportedTypedArrayConstructors>([
  ['float32', Float32Array],
  ['uint8', Uint8Array],
  ['int8', Int8Array],
  ['uint16', Uint16Array],
  ['int16', Int16Array],
  ['int32', Int32Array],
  ['bool', Uint8Array],
  ['float64', Float64Array],
  ['uint32', Uint32Array],
]);

// a runtime map that maps type string to TypedArray constructor. Should match Tensor.DataTypeMap.
const NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP = new Map<SupportedTypedArrayConstructors, Tensor.Type>([
  [Float32Array, 'float32'],
  [Uint8Array, 'uint8'],
  [Int8Array, 'int8'],
  [Uint16Array, 'uint16'],
  [Int16Array, 'int16'],
  [Int32Array, 'int32'],
  [Float64Array, 'float64'],
  [Uint32Array, 'uint32'],
]);

if (isBigInt64ArrayAvailable) {
  NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP.set('int64', BigInt64Array);
  NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP.set(BigInt64Array, 'int64');
}
if (isBigUint64ArrayAvailable) {
  NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP.set('uint64', BigUint64Array);
  NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP.set(BigUint64Array, 'uint64');
}

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

export class Tensor implements Tensor {
  //#region constructors
  constructor(type: Tensor.Type, data: Tensor.DataType|readonly number[]|readonly boolean[], dims?: readonly number[]);
  constructor(data: Tensor.DataType|readonly boolean[], dims?: readonly number[]);
  constructor(
      arg0: Tensor.Type|Tensor.DataType|readonly boolean[], arg1?: Tensor.DataType|readonly number[]|readonly boolean[],
      arg2?: readonly number[]) {
    let type: Tensor.Type;
    let data: Tensor.DataType;
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
          throw new TypeError(`Unsupported tensor type: ${arg0}.`);
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
        const mappedType =
            NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP.get(arg0.constructor as SupportedTypedArrayConstructors);
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
  readonly type: Tensor.Type;
  readonly data: Tensor.DataType;
  readonly size: number;
  //#endregion

  //#region tensor utilities
  reshape(dims: readonly number[]): Tensor {
    return new Tensor(this.type, this.data, dims);
  }
  //#endregion
}
