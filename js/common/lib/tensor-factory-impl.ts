// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorFromGpuBufferOptions, TensorFromImageBitmapOptions, TensorFromImageDataOptions, TensorFromImageElementOptions, TensorFromTextureOptions, TensorFromUrlOptions} from './tensor-factory.js';
import {Tensor} from './tensor-impl.js';
import {Tensor as TensorInterface} from './tensor.js';

/**
 * implementation of Tensor.fromImage().
 */
export const tensorFromImage = async(
    image: ImageData|HTMLImageElement|ImageBitmap|string,
    options?: TensorFromImageDataOptions|TensorFromImageElementOptions|TensorFromImageBitmapOptions|
    TensorFromUrlOptions): Promise<Tensor> => {
  // checking the type of image object
  const isImageElement = typeof HTMLImageElement !== 'undefined' && image instanceof HTMLImageElement;
  const isImageData = typeof ImageData !== 'undefined' && image instanceof ImageData;
  const isImageBitmap = typeof ImageBitmap !== 'undefined' && image instanceof ImageBitmap;
  const isString = typeof image === 'string';

  const channels = options?.tensorFormat?.length ?? 3;
  const type = options?.dataType ?? 'float32';
  if (isImageElement) {
    const width = options?.resizedWidth ?? image.width ?? image.naturalWidth;
    const height = options?.resizedHeight ?? image.height ?? image.naturalHeight;
    const dims = options?.tensorLayout === 'NCHW' ? [1, channels, height, width] : [1, height, width, channels];
    return new Tensor(
        {location: 'pending', input: 'image-element', output: options?.location ?? 'cpu', data: image, dims, type});
  } else if (isImageData) {
    const width = options?.resizedWidth ?? image.width;
    const height = options?.resizedHeight ?? image.height;
    const channels = options?.tensorFormat?.length ?? 3;
    const dims = options?.tensorLayout === 'NCHW' ? [1, channels, height, width] : [1, height, width, channels];
    return new Tensor({location: 'pending', input: 'image-data', output: 'cpu', data: image, dims, type});
  } else if (isImageBitmap) {
    const width = options?.resizedWidth ?? image.width;
    const height = options?.resizedHeight ?? image.height;
    const channels = options?.tensorFormat?.length ?? 3;
    const dims = options?.tensorLayout === 'NCHW' ? [1, channels, height, width] : [1, height, width, channels];
    return new Tensor({location: 'pending', input: 'image-bitmap', output: 'gpu-buffer', data: image, dims, type});
  } else if (isString) {
    // TODO: better support or deprecate for URL
    throw new Error('Not implemented');
  } else {
    throw new Error('Input data provided is not supported - aborted tensor creation');
  }
};

/**
 * implementation of Tensor.fromTexture().
 */
export const tensorFromTexture = <T extends TensorInterface.TextureDataTypes>(
    texture: TensorInterface.TextureType, options: TensorFromTextureOptions<T>): Tensor => {
  const {width, height, download, dispose} = options;
  // Always assume RGBAF32. TODO: support different texture format
  const dims = [1, height, width, 4];
  return new Tensor({location: 'texture', type: 'float32', texture, dims, download, dispose});
};

/**
 * implementation of Tensor.fromGpuBuffer().
 */
export const tensorFromGpuBuffer = <T extends TensorInterface.GpuBufferDataTypes>(
    gpuBuffer: TensorInterface.GpuBufferType, options: TensorFromGpuBufferOptions<T>): Tensor => {
  const {dataType, dims, download, dispose} = options;
  return new Tensor({location: 'gpu-buffer', type: dataType ?? 'float32', gpuBuffer, dims, download, dispose});
};

/**
 * implementation of Tensor.fromPinnedBuffer().
 */
export const tensorFromPinnedBuffer = <T extends TensorInterface.CpuPinnedDataTypes>(
    type: T, buffer: TensorInterface.DataTypeMap[T], dims?: readonly number[]): Tensor =>
    new Tensor({location: 'cpu-pinned', type, data: buffer, dims: dims ?? [buffer.length]});
