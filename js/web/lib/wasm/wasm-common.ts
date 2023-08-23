// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from 'onnxruntime-common';

// This file includes common definitions. They do NOT have dependency on the WebAssembly instance.

/**
 * Copied from ONNX definition. Use this to drop dependency 'onnx_proto' to decrease compiled .js file size.
 */
export const enum DataType {
  undefined = 0,
  float = 1,
  uint8 = 2,
  int8 = 3,
  uint16 = 4,
  int16 = 5,
  int32 = 6,
  int64 = 7,
  string = 8,
  bool = 9,
  float16 = 10,
  double = 11,
  uint32 = 12,
  uint64 = 13,
  complex64 = 14,
  complex128 = 15,
  bfloat16 = 16
}

/**
 * Map string tensor data to enum value
 */
export const tensorDataTypeStringToEnum = (type: string): DataType => {
  switch (type) {
    case 'int8':
      return DataType.int8;
    case 'uint8':
      return DataType.uint8;
    case 'bool':
      return DataType.bool;
    case 'int16':
      return DataType.int16;
    case 'uint16':
      return DataType.uint16;
    case 'int32':
      return DataType.int32;
    case 'uint32':
      return DataType.uint32;
    case 'float16':
      return DataType.float16;
    case 'float32':
      return DataType.float;
    case 'float64':
      return DataType.double;
    case 'string':
      return DataType.string;
    case 'int64':
      return DataType.int64;
    case 'uint64':
      return DataType.uint64;

    default:
      throw new Error(`unsupported data type: ${type}`);
  }
};

/**
 * Map enum value to string tensor data
 */
export const tensorDataTypeEnumToString = (typeProto: DataType): Tensor.Type => {
  switch (typeProto) {
    case DataType.int8:
      return 'int8';
    case DataType.uint8:
      return 'uint8';
    case DataType.bool:
      return 'bool';
    case DataType.int16:
      return 'int16';
    case DataType.uint16:
      return 'uint16';
    case DataType.int32:
      return 'int32';
    case DataType.uint32:
      return 'uint32';
    case DataType.float16:
      return 'float16';
    case DataType.float:
      return 'float32';
    case DataType.double:
      return 'float64';
    case DataType.string:
      return 'string';
    case DataType.int64:
      return 'int64';
    case DataType.uint64:
      return 'uint64';

    default:
      throw new Error(`unsupported data type: ${typeProto}`);
  }
};

/**
 * get tensor element size in bytes by the given data type
 * @returns size in integer or undefined if the data type is not supported
 */
export const getTensorElementSize = (dateType: number): number|
    undefined => [undefined, 4, 1, 1, 2, 2, 4, 8, undefined, 1, 2, 8, 4, 8, undefined, undefined, undefined][dateType];

/**
 * get typed array constructor by the given tensor type
 */
export const tensorTypeToTypedArrayConstructor = (type: Tensor.Type): Float32ArrayConstructor|Uint8ArrayConstructor|
    Int8ArrayConstructor|Uint16ArrayConstructor|Int16ArrayConstructor|Int32ArrayConstructor|BigInt64ArrayConstructor|
    Uint8ArrayConstructor|Float64ArrayConstructor|Uint32ArrayConstructor|BigUint64ArrayConstructor => {
      switch (type) {
        case 'float16':
          return Uint16Array;
        case 'float32':
          return Float32Array;
        case 'uint8':
          return Uint8Array;
        case 'int8':
          return Int8Array;
        case 'uint16':
          return Uint16Array;
        case 'int16':
          return Int16Array;
        case 'int32':
          return Int32Array;
        case 'bool':
          return Uint8Array;
        case 'float64':
          return Float64Array;
        case 'uint32':
          return Uint32Array;
        case 'int64':
          return BigInt64Array;
        case 'uint64':
          return BigUint64Array;
        default:
          throw new Error(`unsupported type: ${type}`);
      }
    };

/**
 * Map string log level to integer value
 */
export const logLevelStringToEnum = (logLevel?: 'verbose'|'info'|'warning'|'error'|'fatal'): number => {
  switch (logLevel) {
    case 'verbose':
      return 0;
    case 'info':
      return 1;
    case 'warning':
      return 2;
    case 'error':
      return 3;
    case 'fatal':
      return 4;
    default:
      throw new Error(`unsupported logging level: ${logLevel}`);
  }
};
