import {Tensor, TypedTensor} from './tensor';

interface Properties {
  /**
   * Get the number of elements in the tensor.
   */
  readonly size: number;
}

interface ShapeUtils {
  /**
   * Create a new tensor with the same data buffer and specified dims.
   * @param dims New dimensions. Size should match the old one.
   */
  reshape(dims: ReadonlyArray<number>): Tensor;
}

export interface TypedShapeUtils<T extends Tensor.Type> extends ShapeUtils {
  reshape(dims: ReadonlyArray<number>): TypedTensor<T>;
}

// TODO: add more tensor utilities
export interface TensorUtils extends Properties, ShapeUtils {}
export interface TypedTensorUtils<T extends Tensor.Type> extends Properties, TypedShapeUtils<T> {}