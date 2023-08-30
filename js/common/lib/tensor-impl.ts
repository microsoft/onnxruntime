// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {tensorToDataURL, tensorToImageData} from './tensor-conversion-impl.js';
import {TensorToDataUrlOptions, TensorToImageDataOptions} from './tensor-conversion.js';
import {tensorFromImage} from './tensor-factory-impl.js';
import {TensorFromImageBitmapOptions, TensorFromImageDataOptions, TensorFromImageElementOptions, TensorFromUrlOptions} from './tensor-factory.js';
import {calculateSize, tensorReshape} from './tensor-utils-impl.js';
import {Tensor as TensorInterface} from './tensor.js';

type TensorType = TensorInterface.Type;
type TensorDataType = TensorInterface.DataType;

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
  ['float16', Uint16Array],
  ['int16', Int16Array],
  ['int32', Int32Array],
  ['bool', Uint8Array],
  ['float64', Float64Array],
  ['uint32', Uint32Array],
]);

// a runtime map that maps type string to TypedArray constructor. Should match Tensor.DataTypeMap.
const NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP = new Map<SupportedTypedArrayConstructors, TensorType>([
  [Float32Array, 'float32'],
  [Uint8Array, 'uint8'],
  [Int8Array, 'int8'],
  [Uint16Array, 'uint16'],
  [Int16Array, 'int16'],
  [Int32Array, 'int32'],
  [Float64Array, 'float64'],
  [Uint32Array, 'uint32'],
]);

// the following code allows delaying execution of BigInt checking. This allows lazy initialization for
// NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP and NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP, which allows BigInt polyfill
// if available.
let isBigIntChecked = false;
const checkBigInt = () => {
  if (!isBigIntChecked) {
    isBigIntChecked = true;
    const isBigInt64ArrayAvailable = typeof BigInt64Array !== 'undefined' && typeof BigInt64Array.from === 'function';
    const isBigUint64ArrayAvailable =
        typeof BigUint64Array !== 'undefined' && typeof BigUint64Array.from === 'function';

    if (isBigInt64ArrayAvailable) {
      NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP.set('int64', BigInt64Array);
      NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP.set(BigInt64Array, 'int64');
    }
    if (isBigUint64ArrayAvailable) {
      NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP.set('uint64', BigUint64Array);
      NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP.set(BigUint64Array, 'uint64');
    }
  }
};


export class Tensor implements TensorInterface {
  // #region constructors
  constructor(type: TensorType, data: TensorDataType|readonly number[]|readonly boolean[], dims?: readonly number[]);
  constructor(data: TensorDataType|readonly boolean[], dims?: readonly number[]);
  constructor(
      arg0: TensorType|TensorDataType|readonly boolean[], arg1?: TensorDataType|readonly number[]|readonly boolean[],
      arg2?: readonly number[]) {
    checkBigInt();

    let type: TensorType;
    let data: TensorDataType;
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
          if (arg0 === 'float16') {
            // Throw error here because when user try to use number array as data,
            // e.g. new Tensor('float16', [1, 2, 3, 4], dims)), it will actually call
            // Uint16Array.from(arg1) which generates wrong data.
            throw new TypeError(
                'Creating a float16 tensor from number array is not supported. Please use Uint16Array as data.');
          } else if (arg0 === 'uint64' || arg0 === 'int64') {
            // use 'as any' here because:
            // 1. TypeScript's check on type of 'Array.isArray()' does not work with readonly arrays.
            // see https://github.com/microsoft/TypeScript/issues/17002
            // 2. TypeScript's check on union type of '(BigInt64ArrayConstructor|BigUint64ArrayConstructor).from()' does
            // not accept parameter mapFn.
            // 3. parameters of 'SupportedTypedArrayConstructors.from()' does not match the requirement of the union
            // type.

            // assume 'arg1' is of type "readonly number[]|readonly bigint[]" here.

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            data = (typedArrayConstructor as any).from(arg1, BigInt);
          } else {
            // assume 'arg1' is of type "readonly number[]" here.

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            data = (typedArrayConstructor as any).from(arg1);
          }
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
  // #endregion

  // #region factory
  static async fromImage(imageData: ImageData, options?: TensorFromImageDataOptions): Promise<Tensor>;
  static async fromImage(imageElement: HTMLImageElement, options?: TensorFromImageElementOptions): Promise<Tensor>;
  static async fromImage(bitmap: ImageBitmap, options: TensorFromImageBitmapOptions): Promise<Tensor>;
  static async fromImage(urlSource: string, options?: TensorFromUrlOptions): Promise<Tensor>;

  static async fromImage(
      image: ImageData|HTMLImageElement|ImageBitmap|string,
      options?: TensorFromImageDataOptions|TensorFromImageElementOptions|TensorFromImageBitmapOptions|
      TensorFromUrlOptions): Promise<Tensor> {
    return tensorFromImage(image, options);
  }
  // #endregion

  // #region conversions
  toDataURL(options?: TensorToDataUrlOptions): string {
    return tensorToDataURL(this, options);
  }

  toImageData(options?: TensorToImageDataOptions): ImageData {
    return tensorToImageData(this, options);
  }
  // #endregion

  // #region fields
  readonly dims: readonly number[];
  readonly type: TensorType;
  readonly data: TensorDataType;
  readonly size: number;
  // #endregion

  // #region tensor utilities
  reshape(dims: readonly number[]): Tensor {
    return tensorReshape(this, dims);
  }
  // #endregion
}
