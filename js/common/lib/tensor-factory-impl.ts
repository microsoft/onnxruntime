// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorFromGpuBufferOptions, TensorFromTextureOptions} from './tensor-factory.js';
import {Tensor} from './tensor-impl.js';
import {Tensor as TensorInterface} from './tensor.js';

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
