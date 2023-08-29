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
 * supported data types for constructing a tensor from a pinned CPU buffer
 */
export type CpuPinnedDataTypes = Exclude<Tensor.Type, 'string'>;

/**
 * represent the parameter for constructing a tensor from a pinned CPU buffer
 */
export interface CpuPinnedConstructorParameters<T extends CpuPinnedDataTypes = CpuPinnedDataTypes> extends
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
 * supported data types for constructing a tensor from a WebGL texture
 */
export type TextureDataTypes = 'float32';

/**
 * represent the parameter for constructing a tensor from a WebGL texture
 */
export interface TextureConstructorParameters<T extends TextureDataTypes = TextureDataTypes> extends
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
 * supported data types for constructing a tensor from a WebGPU buffer
 */
export type GpuBufferDataTypes = 'float32'|'int32';

/**
 * represent the parameter for constructing a tensor from a WebGPU buffer
 */
export interface GpuBufferConstructorParameters<T extends GpuBufferDataTypes = GpuBufferDataTypes> extends
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

// the following region contains type definitions of each individual options.
// the tensor factory functions use a composition of those options as the parameter type.

// #region Options fields

export interface OptionsFormat {
  /**
   * Describes the image format represented in RGBA color space.
   */
  format?: ImageFormat;
}

export interface OptionsTensorFormat {
  /**
   * Describes the image format of the tensor.
   *
   * NOTE: this is different from option 'format'. While option 'format' represents the original image, 'tensorFormat'
   * represents the target format of the tensor. A transpose will be performed if they are different.
   */
  tensorFormat?: ImageFormat;
}

export interface OptionsTensorDataType {
  /**
   * Describes the data type of the tensor.
   */
  dataType?: 'float32'|'uint8';
}

export interface OptionsTensorLayout {
  /**
   * Describes the tensor layout when representing data of one or more image(s).
   */
  tensorLayout?: ImageTensorLayout;
}

export interface OptionsDimensions {
  /**
   * Describes the image height in pixel
   */
  height?: number;
  /**
   * Describes the image width in pixel
   */
  width?: number;
}

export interface OptionResizedDimensions {
  /**
   * Describes the resized height. If omitted, original height will be used.
   */
  resizedHeight?: number;
  /**
   * Describes resized width - can be accessed via tensor dimensions as well
   */
  resizedWidth?: number;
}

export interface OptionsNormalizationParameters {
  /**
   * Describes normalization parameters when preprocessing the image as model input.
   *
   * Data element are ranged from 0 to 255.
   */
  norm?: {
    /**
     * The 'bias' value for image normalization.
     * - If omitted, use default value 0.
     * - If it's a single number, apply to each channel
     * - If it's an array of 3 or 4 numbers, apply element-wise. Number of elements need to match the number of channels
     * for the corresponding image format
     */
    bias?: number|[number, number, number]|[number, number, number, number];
    /**
     * The 'mean' value for image normalization.
     * - If omitted, use default value 255.
     * - If it's a single number, apply to each channel
     * - If it's an array of 3 or 4 numbers, apply element-wise. Number of elements need to match the number of channels
     * for the corresponding image format
     */
    mean?: number | [number, number, number] | [number, number, number, number];
  };
}

// #endregion

// #region Options composition

export interface TensorFromImageDataOptions extends OptionResizedDimensions, OptionsTensorFormat, OptionsTensorLayout,
                                                    OptionsTensorDataType, OptionsNormalizationParameters {}

export interface TensorFromImageElementOptions extends OptionResizedDimensions, OptionsTensorFormat,
                                                       OptionsTensorLayout, OptionsTensorDataType,
                                                       OptionsNormalizationParameters {}

export interface TensorFromUrlOptions extends OptionsDimensions, OptionResizedDimensions, OptionsTensorFormat,
                                              OptionsTensorLayout, OptionsTensorDataType,
                                              OptionsNormalizationParameters {}

export interface TensorFromImageBitmapOptions extends OptionResizedDimensions, OptionsTensorFormat, OptionsTensorLayout,
                                                      OptionsTensorDataType, OptionsNormalizationParameters {}

export interface TensorFromTextureOptions<T extends TextureDataTypes> extends
    Required<OptionsDimensions>, OptionsFormat, GpuResourceConstructorParameters<T>/* TODO: add more */ {}

export interface TensorFromGpuBufferOptions<T extends GpuBufferDataTypes> extends Pick<Tensor, 'dims'>,
                                                                                  GpuResourceConstructorParameters<T> {
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
   * create a tensor from an ImageData object
   *
   * @param imageData - the ImageData object to create tensor from
   * @param options - An optional object representing options for creating tensor from ImageData.
   *
   * The following default settings will be applied:
   * - `tensorFormat`: `'RGB'`
   * - `tensorLayout`: `'NCHW'`
   * - `dataType`: `'float32'`
   * @returns A promise that resolves to a tensor object
   */
  fromImage(imageData: ImageData, options?: TensorFromImageDataOptions):
      Promise<TypedTensor<'float32'>|TypedTensor<'uint8'>>;

  /**
   * create a tensor from a HTMLImageElement object
   *
   * @param imageElement - the HTMLImageElement object to create tensor from
   * @param options - An optional object representing options for creating tensor from HTMLImageElement.
   *
   * The following default settings will be applied:
   * - `tensorFormat`: `'RGB'`
   * - `tensorLayout`: `'NCHW'`
   * - `dataType`: `'float32'`
   * @returns A promise that resolves to a tensor object
   */
  fromImage(imageElement: HTMLImageElement, options?: TensorFromImageElementOptions):
      Promise<TypedTensor<'float32'>|TypedTensor<'uint8'>>;

  /**
   * create a tensor from URL
   *
   * @param urlSource - a string as a URL to the image or a data URL containing the image data.
   * @param options - An optional object representing options for creating tensor from URL.
   *
   * The following default settings will be applied:
   * - `tensorFormat`: `'RGB'`
   * - `tensorLayout`: `'NCHW'`
   * - `dataType`: `'float32'`
   * @returns A promise that resolves to a tensor object
   */
  fromImage(urlSource: string, options?: TensorFromUrlOptions): Promise<TypedTensor<'float32'>|TypedTensor<'uint8'>>;

  /**
   * create a tensor from an ImageBitmap object
   *
   * @param bitMap - the ImageBitmap object to create tensor from
   * @param options - An optional object representing options for creating tensor from URL.
   *
   * The following default settings will be applied:
   * - `tensorFormat`: `'RGB'`
   * - `tensorLayout`: `'NCHW'`
   * - `dataType`: `'float32'`
   * @returns A promise that resolves to a tensor object
   */
  fromImage(bitmap: ImageBitmap, options: TensorFromImageBitmapOptions):
      Promise<TypedTensor<'float32'>|TypedTensor<'uint8'>>;

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
  fromTexture<T extends TextureDataTypes = 'float32'>(
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
  fromGpuBuffer<T extends GpuBufferDataTypes = 'float32'>(
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
