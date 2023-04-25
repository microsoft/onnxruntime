// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

export declare namespace Tensor {
  export interface DataTypeMap {
    bool: Uint8Array;
    float32: Float32Array;
    float64: Float64Array;
    string: string[];
    int8: Int8Array;
    uint8: Uint8Array;
    int16: Int16Array;
    uint16: Uint16Array;
    int32: Int32Array;
    uint32: Uint32Array;
    int64: BigInt64Array;
    uint64: BigUint64Array;
  }

  export type DataType = keyof DataTypeMap;

  export type StringType = Tensor.DataTypeMap['string'];
  export type BooleanType = Tensor.DataTypeMap['bool'];
  export type IntegerType = Tensor.DataTypeMap['int8']|Tensor.DataTypeMap['uint8']|Tensor.DataTypeMap['int16']|
                            Tensor.DataTypeMap['uint16']|Tensor.DataTypeMap['int32']|Tensor.DataTypeMap['uint32']|
                            Tensor.DataTypeMap['int64']|Tensor.DataTypeMap['uint64'];
  export type FloatType = Tensor.DataTypeMap['float32']|Tensor.DataTypeMap['float64'];
  export type NumberType = BooleanType|IntegerType|FloatType;

  export type Id = number;
}

export const sizeof = (type: Tensor.DataType): number => {
  switch (type) {
    case 'bool':
    case 'int8':
    case 'uint8':
      return 1;
    case 'int16':
    case 'uint16':
      return 2;
    case 'int32':
    case 'uint32':
    case 'float32':
      return 4;
    case 'int64':
    case 'uint64':
    case 'float64':
      return 8;
    default:
      throw new Error(`cannot calculate sizeof() on type ${type}`);
  }
};

const dataviewConstructor = (type: Tensor.DataType) => {
  switch (type) {
    case 'bool':
    case 'uint8':
      return Uint8Array;
    case 'int8':
      return Int8Array;
    case 'int16':
      return Int16Array;
    case 'uint16':
      return Uint16Array;
    case 'int32':
      return Int32Array;
    case 'uint32':
      return Uint32Array;
    case 'int64':
      return BigInt64Array;
    case 'uint64':
      return BigUint64Array;
    case 'float32':
      return Float32Array;
    case 'float64':
      return Float64Array;
    default:
      // should never run to here
      throw new Error('unspecified error');
  }
};

export const createView = (dataBuffer: ArrayBuffer, type: Tensor.DataType): Int32Array|Uint32Array|BigInt64Array|
    BigUint64Array|Uint8Array|Float32Array|Float64Array|Int8Array|Int16Array|Uint16Array =>
        new (dataviewConstructor(type))(dataBuffer);

/**
 * a TensorView does not own the data.
 */
export interface TensorView {
  readonly data: number;
  readonly dataType: number;
  readonly dims: readonly number[];

  /**
   * get a Float32Array data view of the tensor data. tensor data must be on CPU.
   */
  getFloat32Array(): Float32Array;

  /**
   * create a new tensor view with the same data but different dimensions.
   */
  reshape(newDims: readonly number[]): TensorView;
}
