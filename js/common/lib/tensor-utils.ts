// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor, TypedTensor} from './tensor';
import {ConversionUtils} from './tensor-conversion';

interface Properties {
  /**
   * Get the number of elements in the tensor.
   */
  readonly size: number;
}

export interface TypedShapeUtils<T extends Tensor.Type> {
  /**
   * Create a new tensor with the same data buffer and specified dims.
   *
   * @param dims - New dimensions. Size should match the old one.
   */
  reshape(dims: readonly number[]): TypedTensor<T>;
}

/**
 * interface `TensorUtils` includes all utility members that does not use the type parameter from their signature.
 */
export interface TensorUtils extends Properties, ConversionUtils {}

/**
 * interface `TypedShapeUtils` includes all utility members that uses the type parameter from their signature.
 */
export interface TypedTensorUtils<T extends Tensor.Type> extends TensorUtils, TypedShapeUtils<T> {}
