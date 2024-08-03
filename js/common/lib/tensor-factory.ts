// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor, TypedTensor} from './tensor.js';

export type ImageFormat = 'RGB'|'RGBA'|'BGR'|'RBG';
export type ImageTensorLayout = 'NHWC'|'NCHW';

// the following region contains type definitions for constructing tensor from a specific location.

// #region types for constructing a tensor from a specific location

/**
 * represent common properties of the parameter for constructing a tensor from a specific location.
 */
interface CommonConstructorParameters<T> extends Pick<Tensor, 'dims'> {
  /**
   * Specify the data type of the tensor.
   */
  readonly type: T;
}

/**
 * represent the parameter for constructing a tensor from a GPU resource.
 */
interface GpuResourceConstructorParameters<T extends Tensor.Type> {
  /**
   * an optional callback function to download data from GPU to CPU.
   *
   * If not provided, the tensor treat the GPU data as external resource.
   */
  download?(): Promise<Tensor.DataTypeMap[T]>;

  /**
   * an optional callback function that will be called when the tensor is disposed.
   *
   * If not provided, the tensor treat the GPU data as external resource.
   */
  dispose?(): void;
}

/**
 * represent the parameter for constructing a tensor from a pinned CPU buffer
 */
export interface CpuPinnedConstructorParameters<T extends Tensor.CpuPinnedDataTypes = Tensor.CpuPinnedDataTypes> extends
    CommonConstructorParameters<T> {
  /**
   * Specify the location of the data to be 'cpu-pinned'.
   */
  readonly location: 'cpu-pinned';
  /**
   * Specify the CPU pinned buffer that holds the tensor data.
   */
  readonly data: Tensor.DataTypeMap[T];
}

/**
 * represent the parameter for constructing a tensor from a WebGL texture
 */
export interface TextureConstructorParameters<T extends Tensor.TextureDataTypes = Tensor.TextureDataTypes> extends
    CommonConstructorParameters<T>, GpuResourceConstructorParameters<T> {
  /**
   * Specify the location of the data to be 'texture'.
   */
  readonly location: 'texture';
  /**
   * Specify the WebGL texture that holds the tensor data.
   */
  readonly texture: Tensor.TextureType;
}

/**
 * represent the parameter for constructing a tensor from a WebGPU buffer
 */
export interface GpuBufferConstructorParameters<T extends Tensor.GpuBufferDataTypes = Tensor.GpuBufferDataTypes> extends
    CommonConstructorParameters<T>, GpuResourceConstructorParameters<T> {
  /**
   * Specify the location of the data to be 'gpu-buffer'.
   */
  readonly location: 'gpu-buffer';
  /**
   * Specify the WebGPU buffer that holds the tensor data.
   */
  readonly gpuBuffer: Tensor.GpuBufferType;
}

// #endregion

// #region Options composition

export interface TensorFromTextureOptions<T extends Tensor.TextureDataTypes> extends
    GpuResourceConstructorParameters<T>/* TODO: add more */ {
  /**
   * Describes the image height in pixel
   */
  height: number;
  /**
   * Describes the image width in pixel
   */
  width: number;
  /**
   * Describes the image format represented in RGBA color space.
   */
  format?: ImageFormat;
}

export interface TensorFromGpuBufferOptions<T extends Tensor.GpuBufferDataTypes> extends
    Pick<Tensor, 'dims'>, GpuResourceConstructorParameters<T> {
  /**
   * Describes the data type of the tensor.
   */
  dataType?: T;
}

// #endregion

/**
 * type TensorFactory defines the factory functions of 'Tensor' to create tensor instances from existing data or
 * resources.
 */
export interface TensorFactory {
  /**
   * create a tensor from a WebGL texture
   *
   * @param texture - the WebGLTexture object to create tensor from
   * @param options - An optional object representing options for creating tensor from WebGL texture.
   *
   * The options include following properties:
   * - `width`: the width of the texture. Required.
   * - `height`: the height of the texture. Required.
   * - `format`: the format of the texture. If omitted, assume 'RGBA'.
   * - `download`: an optional function to download the tensor data from GPU to CPU. If omitted, the GPU data
   * will not be able to download. Usually, this is provided by a GPU backend for the inference outputs. Users don't
   * need to provide this function.
   * - `dispose`: an optional function to dispose the tensor data on GPU. If omitted, the GPU data will not be disposed.
   * Usually, this is provided by a GPU backend for the inference outputs. Users don't need to provide this function.
   *
   * @returns a tensor object
   */
  fromTexture<T extends Tensor.TextureDataTypes = 'float32'>(
      texture: Tensor.TextureType, options: TensorFromTextureOptions<T>): TypedTensor<'float32'>;

  /**
   * create a tensor from a WebGPU buffer
   *
   * @param buffer - the GPUBuffer object to create tensor from
   * @param options - An optional object representing options for creating tensor from WebGPU buffer.
   *
   * The options include following properties:
   * - `dataType`: the data type of the tensor. If omitted, assume 'float32'.
   * - `dims`: the dimension of the tensor. Required.
   * - `download`: an optional function to download the tensor data from GPU to CPU. If omitted, the GPU data
   * will not be able to download. Usually, this is provided by a GPU backend for the inference outputs. Users don't
   * need to provide this function.
   * - `dispose`: an optional function to dispose the tensor data on GPU. If omitted, the GPU data will not be disposed.
   * Usually, this is provided by a GPU backend for the inference outputs. Users don't need to provide this function.
   *
   * @returns a tensor object
   */
  fromGpuBuffer<T extends Tensor.GpuBufferDataTypes>(
      buffer: Tensor.GpuBufferType, options: TensorFromGpuBufferOptions<T>): TypedTensor<T>;

  /**
   * create a tensor from a pre-allocated buffer. The buffer will be used as a pinned buffer.
   *
   * @param type - the tensor element type.
   * @param buffer - a TypedArray corresponding to the type.
   * @param dims - specify the dimension of the tensor. If omitted, a 1-D tensor is assumed.
   *
   * @returns a tensor object
   */
  fromPinnedBuffer<T extends Exclude<Tensor.Type, 'string'>>(
      type: T, buffer: Tensor.DataTypeMap[T], dims?: readonly number[]): TypedTensor<T>;
}
