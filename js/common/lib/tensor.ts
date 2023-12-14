// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorFactory} from './tensor-factory.js';
import {Tensor as TensorImpl} from './tensor-impl.js';
import {TypedTensorUtils} from './tensor-utils.js';

/* eslint-disable @typescript-eslint/no-redeclare */

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
   *
   * If the data is not on CPU (eg. it's in the form of WebGL texture or WebGPU buffer), throw error.
   */
  readonly data: Tensor.DataTypeMap[T];
  /**
   * Get the location of the data.
   */
  readonly location: Tensor.DataLocation;
  /**
   * Get the WebGL texture that holds the tensor data.
   *
   * If the data is not on GPU as WebGL texture, throw error.
   */
  readonly texture: Tensor.TextureType;
  /**
   * Get the WebGPU buffer that holds the tensor data.
   *
   * If the data is not on GPU as WebGPU buffer, throw error.
   */
  readonly gpuBuffer: Tensor.GpuBufferType;

  /**
   * Get the buffer data of the tensor.
   *
   * If the data is on CPU, returns the data immediately.
   * If the data is on GPU, downloads the data and returns the promise.
   *
   * @param releaseData - whether release the data on GPU. Ignore if data is already on CPU.
   */
  getData(releaseData?: boolean): Promise<Tensor.DataTypeMap[T]>;

  /**
   * Dispose the tensor data.
   *
   * If the data is on CPU, remove its internal reference to the underlying data.
   * If the data is on GPU, release the data on GPU.
   *
   * After calling this function, the tensor is considered no longer valid. Its location will be set to 'none'.
   */
  dispose(): void;
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
    float16: Uint16Array;  // Keep using Uint16Array until we have a concrete solution for float 16.
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
    int64: bigint;
    string: string;
    bool: boolean;
    float16: number;  // Keep using Uint16Array until we have a concrete solution for float 16.
    float64: number;
    uint32: number;
    uint64: bigint;
    // complex64: never;
    // complex128: never;
    // bfloat16: never;
  }

  type DataType = DataTypeMap[Type];
  type ElementType = ElementTypeMap[Type];

  /**
   * supported data types for constructing a tensor from a pinned CPU buffer
   */
  export type CpuPinnedDataTypes = Exclude<Tensor.Type, 'string'>;

  /**
   * type alias for WebGL texture
   */
  export type TextureType = WebGLTexture;

  /**
   * supported data types for constructing a tensor from a WebGL texture
   */
  export type TextureDataTypes = 'float32';

  /**
   * type alias for WebGPU buffer
   *
   * The reason why we don't use type "GPUBuffer" defined in webgpu.d.ts from @webgpu/types is because "@webgpu/types"
   * requires "@types/dom-webcodecs" as peer dependency when using TypeScript < v5.1 and its version need to be chosen
   * carefully according to the TypeScript version being used. This means so far there is not a way to keep every
   * TypeScript version happy. It turns out that we will easily broke users on some TypeScript version.
   *
   * for more info see https://github.com/gpuweb/types/issues/127
   */
  export type GpuBufferType = {size: number; mapState: 'unmapped' | 'pending' | 'mapped'};

  /**
   * supported data types for constructing a tensor from a WebGPU buffer
   */
  export type GpuBufferDataTypes = 'float32'|'float16'|'int32'|'int64'|'uint32'|'bool';

  /**
   * represent where the tensor data is stored
   */
  export type DataLocation =|
      /**
       * the data is disposed and became unavailable.
       */
      'none'|
      /**
       * the data is stored in CPU, as either a string array (for 'string' tensor type) or a TypedArray (for numeric
       * tensor type).
       */
      'cpu'|
      /**
       * the numeric tensor data is stored in CPU as TypedArray with pinned buffer.
       */
      'cpu-pinned'|
      /**
       * the numeric tensor data is stored as WebGL texture.
       */
      'texture'|
      /**
       * the numeric tensor data is stored as WebGPU buffer.
       */
      'gpu-buffer'|
      /**
       * the data is pending on one or more pre-processing step(s) before it became available.
       */
      'pending';

  /**
   * represent the data type of a tensor
   */
  export type Type = keyof DataTypeMap;
}

/**
 * Represent multi-dimensional arrays to feed to or fetch from model inferencing.
 */
export interface TypedTensor<T extends Tensor.Type> extends TypedTensorBase<T>, TypedTensorUtils<T> {}
/**
 * Represent multi-dimensional arrays to feed to or fetch from model inferencing.
 */
export interface Tensor extends TypedTensorBase<Tensor.Type>, TypedTensorUtils<Tensor.Type> {}

/**
 * type TensorConstructor defines the constructors of 'Tensor' to create CPU tensor instances.
 */
export interface TensorConstructor {
  // #region CPU tensor - specify element type
  /**
   * Construct a new string tensor object from the given type, data and dims.
   *
   * @param type - Specify the element type.
   * @param data - Specify the CPU tensor data.
   * @param dims - Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(type: 'string', data: Tensor.DataTypeMap['string']|readonly string[],
      dims?: readonly number[]): TypedTensor<'string'>;

  /**
   * Construct a new bool tensor object from the given type, data and dims.
   *
   * @param type - Specify the element type.
   * @param data - Specify the CPU tensor data.
   * @param dims - Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(type: 'bool', data: Tensor.DataTypeMap['bool']|readonly boolean[], dims?: readonly number[]): TypedTensor<'bool'>;

  /**
   * Construct a new 64-bit integer typed tensor object from the given type, data and dims.
   *
   * @param type - Specify the element type.
   * @param data - Specify the CPU tensor data.
   * @param dims - Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new<T extends 'uint64'|'int64'>(
      type: T, data: Tensor.DataTypeMap[T]|readonly bigint[]|readonly number[],
      dims?: readonly number[]): TypedTensor<T>;

  /**
   * Construct a new numeric tensor object from the given type, data and dims.
   *
   * @param type - Specify the element type.
   * @param data - Specify the CPU tensor data.
   * @param dims - Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new<T extends Exclude<Tensor.Type, 'string'|'bool'|'uint64'|'int64'>>(
      type: T, data: Tensor.DataTypeMap[T]|readonly number[], dims?: readonly number[]): TypedTensor<T>;
  // #endregion

  // #region CPU tensor - infer element types

  /**
   * Construct a new float32 tensor object from the given data and dims.
   *
   * @param data - Specify the CPU tensor data.
   * @param dims - Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: Float32Array, dims?: readonly number[]): TypedTensor<'float32'>;

  /**
   * Construct a new int8 tensor object from the given data and dims.
   *
   * @param data - Specify the CPU tensor data.
   * @param dims - Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: Int8Array, dims?: readonly number[]): TypedTensor<'int8'>;

  /**
   * Construct a new uint8 tensor object from the given data and dims.
   *
   * @param data - Specify the CPU tensor data.
   * @param dims - Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: Uint8Array, dims?: readonly number[]): TypedTensor<'uint8'>;

  /**
   * Construct a new uint16 tensor object from the given data and dims.
   *
   * @param data - Specify the CPU tensor data.
   * @param dims - Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: Uint16Array, dims?: readonly number[]): TypedTensor<'uint16'>;

  /**
   * Construct a new int16 tensor object from the given data and dims.
   *
   * @param data - Specify the CPU tensor data.
   * @param dims - Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: Int16Array, dims?: readonly number[]): TypedTensor<'int16'>;

  /**
   * Construct a new int32 tensor object from the given data and dims.
   *
   * @param data - Specify the CPU tensor data.
   * @param dims - Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: Int32Array, dims?: readonly number[]): TypedTensor<'int32'>;

  /**
   * Construct a new int64 tensor object from the given data and dims.
   *
   * @param data - Specify the CPU tensor data.
   * @param dims - Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: BigInt64Array, dims?: readonly number[]): TypedTensor<'int64'>;

  /**
   * Construct a new string tensor object from the given data and dims.
   *
   * @param data - Specify the CPU tensor data.
   * @param dims - Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: readonly string[], dims?: readonly number[]): TypedTensor<'string'>;

  /**
   * Construct a new bool tensor object from the given data and dims.
   *
   * @param data - Specify the CPU tensor data.
   * @param dims - Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: readonly boolean[], dims?: readonly number[]): TypedTensor<'bool'>;

  /**
   * Construct a new float64 tensor object from the given data and dims.
   *
   * @param data - Specify the CPU tensor data.
   * @param dims - Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: Float64Array, dims?: readonly number[]): TypedTensor<'float64'>;

  /**
   * Construct a new uint32 tensor object from the given data and dims.
   *
   * @param data - Specify the CPU tensor data.
   * @param dims - Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: Uint32Array, dims?: readonly number[]): TypedTensor<'uint32'>;

  /**
   * Construct a new uint64 tensor object from the given data and dims.
   *
   * @param data - Specify the CPU tensor data.
   * @param dims - Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: BigUint64Array, dims?: readonly number[]): TypedTensor<'uint64'>;

  // #endregion

  // #region CPU tensor - fall back to non-generic tensor type declaration

  /**
   * Construct a new tensor object from the given type, data and dims.
   *
   * @param type - Specify the element type.
   * @param data - Specify the CPU tensor data.
   * @param dims - Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(type: Tensor.Type, data: Tensor.DataType|readonly number[]|readonly string[]|readonly bigint[]|readonly boolean[],
      dims?: readonly number[]): Tensor;

  /**
   * Construct a new tensor object from the given data and dims.
   *
   * @param data - Specify the CPU tensor data.
   * @param dims - Specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   */
  new(data: Tensor.DataType, dims?: readonly number[]): Tensor;
  // #endregion
}

// eslint-disable-next-line @typescript-eslint/naming-convention
export const Tensor = TensorImpl as (TensorConstructor & TensorFactory);
