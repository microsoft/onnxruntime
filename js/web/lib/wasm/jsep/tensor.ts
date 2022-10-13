// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {ShapeUtil} from './util';

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

type TensorData = Tensor.DataTypeMap[Tensor.DataType];

type DataProvider = (id: Tensor.Id) => TensorData;
type AsyncDataProvider = (id: Tensor.Id) => Promise<TensorData>;

let guid = 0;
const createNewTensorId = () => guid++;


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

export class Tensor {
  /**
   * get the underlying tensor data
   */
  get data(): TensorData {
    if (this.cache === undefined) {
      const data = this.dataProvider!(this.dataId);
      if (data.length !== this.size) {
        throw new Error('Length of data provided by the Data Provider is inconsistent with the dims of this Tensor.');
      }
      this.cache = data;
    }
    return this.cache;
  }

  /**
   * get the underlying string tensor data. Should only use when type is STRING
   */
  get stringData(): Tensor.StringType {
    if (this.type !== 'string') {
      throw new TypeError('data type is not string');
    }

    return this.data as Tensor.StringType;
  }

  /**
   * get the underlying integer tensor data. Should only use when type is one of the following: (UINT8, INT8, UINT16,
   * INT16, INT32, UINT32, BOOL)
   */
  get integerData(): Tensor.IntegerType {
    switch (this.type) {
      case 'uint8':
      case 'int8':
      case 'uint16':
      case 'int16':
      case 'int32':
      case 'uint32':
      case 'int64':
      case 'uint64':
      case 'bool':
        return this.data as Tensor.IntegerType;

      default:
        throw new TypeError(
            'data type is not integer (uint8, int8, uint16, int16, int32, uint32, int64, uint64, bool)');
    }
  }

  /**
   * get the underlying float tensor data. Should only use when type is one of the following: (FLOAT, DOUBLE)
   */
  get floatData(): Tensor.FloatType {
    switch (this.type) {
      case 'float32':
      case 'float64':
        return this.data as Tensor.FloatType;

      default:
        throw new TypeError('data type is not float (float32, float64)');
    }
  }

  /**
   * get the underlying number tensor data. Should only use when type is one of the following: (UINT8, INT8, UINT16,
   * INT16, INT32, UINT32, BOOL, FLOAT, DOUBLE)
   */
  get numberData(): Tensor.NumberType {
    if (this.type !== 'string') {
      return this.data as Tensor.NumberType;
    }
    throw new TypeError('type cannot be non-number (string)');
  }

  /**
   * get the underlying tensor data asynchronously
   */
  async getData(): Promise<TensorData> {
    if (this.cache === undefined) {
      if (this.asyncDataProvider) {
        const data = await this.asyncDataProvider(this.dataId);
        if (data.length !== this.size) {
          throw new Error('Length of data provided by the Data Provider is inconsistent with the dims of this Tensor.');
        }
        this.cache = data;
      } else {
        return this.data;
      }
    }
    return this.cache;
  }

  /**
   * get the number of elements in the tensor
   */
  public readonly size: number;

  private _strides: readonly number[];
  /**
   * get the strides for each dimension
   */
  get strides(): readonly number[] {
    if (!this._strides) {
      this._strides = ShapeUtil.computeStrides(this.dims);
    }
    return this._strides;
  }

  constructor(
      /**
       * get the dimensions of the tensor
       */
      public readonly dims: readonly number[],
      /**
       * get the type of the tensor
       */
      public readonly type: Tensor.DataType, private dataProvider?: DataProvider,
      private asyncDataProvider?: AsyncDataProvider, private cache?: TensorData,
      /**
       * get the data ID that used to map to a tensor data
       */
      public readonly dataId: Tensor.Id = createNewTensorId()) {
    this.size = ShapeUtil.validateDimsAndCalcSize(dims);
    const size = this.size;
    const empty = (dataProvider === undefined && asyncDataProvider === undefined && cache === undefined);

    if (cache !== undefined) {
      if (cache.length !== size) {
        throw new RangeError('Input dims doesn\'t match data length.');
      }
    }

    if (type === 'string') {
      if (cache !== undefined && (!Array.isArray(cache) || !cache.every(i => typeof i === 'string'))) {
        throw new TypeError('cache should be a string array');
      }

      if (empty) {
        this.cache = new Array<string>(size);
      }
    } else {
      if (cache !== undefined) {
        const constructor = dataviewConstructor(type);
        if (!(cache instanceof constructor)) {
          throw new TypeError(`cache should be type ${constructor.name}`);
        }
      }

      if (empty) {
        const buf = new ArrayBuffer(size * sizeof(type));
        this.cache = createView(buf, type);
      }
    }
  }

  /**
   * Construct new Tensor from raw data
   * @param data the raw data object. Should be a string array for 'string' tensor, and the corresponding typed array
   * for other types of tensor.
   * @param dims the dimensions of the tensor
   * @param type the type of the tensor
   */
  static fromData(data: Tensor.DataTypeMap[Tensor.DataType], dims: readonly number[], type: Tensor.DataType): Tensor {
    return new Tensor(dims, type, undefined, undefined, data);
  }
}
