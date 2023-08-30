// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TypedTensor} from './tensor.js';

export type ImageFormat = 'RGB'|'RGBA'|'BGR'|'RBG';
export type ImageTensorLayout = 'NHWC'|'NCHW';

// the following session contains type definitions of each individual options.
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
}
