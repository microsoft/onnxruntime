// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* eslint-disable no-param-reassign */

export class MatMulUtil {
  /**
   * Fix the input shapes for MatMul operation if they need fixing
   * @param dimsA The shape of tensor A. Should be an array of positive integers
   * @param dimsB The shape of tensor B. Should be an array of positive integers
   * @returns A tuple containing the preprocessed input shapes as required by ONNX specifications
   */
  static preprocessInputShapes(dimsA: readonly number[], dimsB: readonly number[]):
      [readonly number[], readonly number[]] {
    // If the first argument is 1-D, it is promoted to a matrix by prepending
    // a 1 to its dimensions. After matrix multiplication the prepended 1 is
    // removed.
    const a = (dimsA.length === 1) ? [1, dimsA[0]] : dimsA;

    // If the second argument is 1-D, it is promoted to a matrix by appending
    // a 1 to its dimensions. After matrix multiplication the appended 1 is
    // removed.
    const b = (dimsB.length === 1) ? [dimsB[0], 1] : dimsB;

    return [a, b];
  }

  /**
   * Fix the output shape computed for MatMul operation if it needs fixing
   * @param outputShape The computed outputShape. Should be an array (atleast of length 2) of positive integers.
   * This will be mutated.
   * @param aRank The rank of tensor A.
   * @param bRank The rank of tensor B.
   */
  static postprocessOutputShape(outputShape: number[], aRank: number, bRank: number): void {
    // Remove prepended dimension if first input is 1d
    if (aRank === 1) {
      // outputShape = outputShape.slice(0, outputShape.length - 2).concat(outputShape.slice(outputShape.length - 1));
      outputShape.splice(outputShape.length - 2, 1);
    }
    // Remove appended dimension if second input is 1d
    if (bRank === 1) {
      outputShape.pop();
    }
  }

  /**
   * Calculate the expected shape when matrix multiplication
   * @param a The shape of tensor A. Should be a tuple of 2 positive integers
   * @param b The shape of tensor B. Should be a tuple of 2 positive integers
   * @returns The expected shape of the result, or undefined if N/A
   */
  static calcMatMulShape(a: [number, number], b: [number, number]): [number, number]|undefined {
    return (a[1] !== b[0]) ? undefined : [a[0], b[1]];
  }
}


export class BroadcastUtil {
  /**
   * Calculate the expected shape when broadcasting 2 tensors
   * @param a The shape of tensor A. Should be an array of positive integers
   * @param b The shape of tensor B. Should be an array of positive integers
   * @param isMatMul Whether the operation is MatMul
   * @returns The expected shape of the result, or undefined if N/A
   */
  static calcShape(adims: readonly number[], bdims: readonly number[], isMatMul = false): readonly number[]|undefined {
    const arank = adims.length;
    const brank = bdims.length;
    if (arank === 0) {
      return bdims;
    }
    if (brank === 0) {
      return adims;
    }
    const crank = Math.max(adims.length, bdims.length);
    const cdims = new Array<number>(crank);

    // calculate the last 2 dimension if it is MatMul
    if (isMatMul) {
      if (arank < 2 || brank < 2) {
        return undefined;
      }
      const cShapeMatMul =
          MatMulUtil.calcMatMulShape([adims[arank - 2], adims[arank - 1]], [bdims[brank - 2], bdims[brank - 1]]);
      if (cShapeMatMul === undefined) {
        return undefined;
      }
      [cdims[crank - 2], cdims[crank - 1]] = cShapeMatMul;
    }

    for (let i = isMatMul ? 3 : 1; i <= crank; i++) {
      const aLen = arank - i < 0 ? 1 : adims[arank - i];
      const bLen = brank - i < 0 ? 1 : bdims[brank - i];

      if (aLen !== bLen && aLen > 1 && bLen > 1) {
        return undefined;
      }
      cdims[crank - i] = Math.max(aLen, bLen);
    }

    return cdims;
  }

  /**
   * Given the indices of a broadcasted tensor, calculate the original indices
   * @param broadcastedIndices The given indices of the broadcasted tensor.
   * @param originalShape The original shape of the tensor before broadcas
   * @returns The calculated indices that maps to the original tensor.
   */
  static index(broadcastedIndices: readonly number[], originalShape: readonly number[]): number[] {
    // NOTE 1: we assume the parameter broadcastedIndices is valid. ie. it should have the same
    // length as the broadcasted shape, and for each dimension the index should
    // not be out of range.
    const originalIndices = new Array(originalShape.length);
    BroadcastUtil.fillIndex(broadcastedIndices, originalShape, originalIndices);
    return originalIndices;
  }

  /**
   * Given the indices of a broadcasted tensor, calculate the original indices
   * @param broadcastedIndices The given indices of the broadcasted tensor.
   * @param originalShape The original shape of the tensor before broadcast
   * @param originalIndices The mapping of broadcastedIndices to the originalIndices (output parameter - will be
   *     mutated).
   */
  static fillIndex(broadcastedIndices: readonly number[], originalShape: readonly number[], originalIndices: number[]):
      void {
    // NOTE 1: we assume the parameter broadcastedIndices is valid. ie. it should have the same length as the
    // broadcasted shape, and for each dimension the index should not be out of range.
    // NOTE 2: we assume the parameter originalIndices has the same length as the originalShape
    const dimOffset = broadcastedIndices.length - originalShape.length;
    for (let i = 0; i < originalShape.length; i++) {
      originalIndices[i] = broadcastedIndices[dimOffset + i] % originalShape[i];
    }
  }

  /**
   * Determine if a shape is unidirectional broadcastable to another shape
   * @param shape The input shape
   * @param finalShape The desired shape after broadcasting
   */
  static isValidBroadcast(shape: readonly number[], finalShape: readonly number[]): boolean {
    // align shape to the right
    const inputRank = shape.length;
    const finalRank = finalShape.length;
    if (inputRank > finalRank) {
      return false;
    }
    for (let i = 1; i <= inputRank; i++) {
      if (shape[inputRank - i] !== 1 && shape[inputRank - i] !== finalShape[finalRank - i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * Determine the broadcasted dims in input shape based on the given output shape.
   * Note that this function only returns the broadcasted dims.
   * @param inputShape The input shape
   * @param outputShape The output shape
   * @returns The broadcasted dims in input shape.
   */
  static getBroadcastDims(inputShape: readonly number[], outputShape: readonly number[]): number[] {
    const inRank = inputShape.length;
    const dims: number[] = [];
    for (let i = 0; i < inRank; i++) {
      const dim = inRank - 1 - i;
      const a = inputShape[dim] || 1;
      const b = outputShape[outputShape.length - 1 - i] || 1;
      if (b > 1 && a === 1) {
        dims.unshift(dim);
      }
    }
    return dims;
  }
}


export class ShapeUtil {
  static size(dims: readonly number[]): number {
    return ShapeUtil.getSizeFromDimensionRange(dims, 0, dims.length);
  }

  // `axis` inclusive
  static sizeFromDimension(dims: readonly number[], axis: number): number {
    if (axis < 0 || axis > dims.length) {
      throw new Error(`invalid dimension of ${axis} for sizeFromDimension as Tensor has ${dims.length} dimensions.`);
    }
    return ShapeUtil.getSizeFromDimensionRange(dims, axis, dims.length);
  }

  // `axis` exclusive
  static sizeToDimension(dims: readonly number[], axis: number): number {
    if (axis < 0 || axis > dims.length) {
      throw new Error(`invalid dimension of ${axis} for sizeToDimension as Tensor has ${dims.length} dimensions.`);
    }
    return ShapeUtil.getSizeFromDimensionRange(dims, 0, axis);
  }

  static getSizeFromDimensionRange(dims: readonly number[], start: number, end: number): number {
    let size = 1;
    for (let i = start; i < end; i++) {
      // safety check as this method is called by multiple other methods requiring size.
      // size cannot be 0 or negative.
      if (dims[i] <= 0) {
        throw new Error(
            // eslint-disable-next-line max-len
            'cannot get valid size from specified dimension range. Most likely the range contains 0 or negative values in them.');
      }
      size *= dims[i];
    }
    return size;
  }

  static computeStrides(dims: readonly number[]): readonly number[] {
    const rank = dims.length;
    if (rank === 0) {
      return [];
    } else if (rank === 1) {
      return [1];
    }
    const strides = new Array(rank);
    strides[rank - 1] = 1;
    strides[rank - 2] = dims[rank - 1];
    for (let i = rank - 3; i >= 0; --i) {
      strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
  }

  static transpose(dims: readonly number[]): readonly number[] {
    const copy = dims.slice();
    return copy.reverse();
  }

  static indicesToOffset(indices: readonly number[], strides: readonly number[], axis?: number): number {
    if (axis === undefined) {
      axis = indices.length;
    }
    let offset = 0;
    for (let i = 0; i < axis; ++i) {
      offset += strides[i] * indices[i];
    }
    return offset;
  }

  static offsetToIndices(offset: number, strides: readonly number[]): readonly number[] {
    const rank = strides.length;
    if (rank === 0) {
      return [];
    } else if (rank === 1) {
      return [offset * strides[0]];
    }
    const indices: number[] = new Array(strides.length);
    for (let i = 0; i < indices.length - 1; ++i) {
      indices[i] = Math.floor(offset / strides[i]);
      offset -= indices[i] * strides[i];
    }
    indices[indices.length - 1] = offset;
    return indices;
  }

  /**
   * normailze axis of range [-r, r) into [0, r).
   */
  static normalizeAxis(axis: number, tensorRank: number): number {
    if (axis < -tensorRank && axis >= tensorRank) {
      throw new Error('unsupported axis for this operation.');
    }
    return axis < 0 ? axis + tensorRank : axis;
  }

  static normalizeAxes(axes: readonly number[], tensorRank: number): number[] {
    return axes.map(x => this.normalizeAxis(x, tensorRank));
  }

  // Increment an index into a tensor (in lexicographic
  // ordering), wrapping around the specified upper_bound.
  /**
   * Increment an index into a tensor (in lexicographic ordering), wrapping around the specified upper_bound.
   * @param index Given index to increment (Will be mutated)
   * @param dims The dimensions of the tensor for which the given index corresponds to
   * @param axisToIncrementOn The 1-indexed axis to increment on. If undefined, axisToIncrementOn == rank
   */
  static incrementIndex(index: number[], dims: readonly number[], axisToIncrementOn?: number): void {
    if (dims.length === 0 || index.length === 0) {
      throw new Error('Index incrementing unsupported for scalar Tensor');
    }
    if (axisToIncrementOn === undefined) {
      axisToIncrementOn = dims.length;
    } else {
      if (axisToIncrementOn <= 0 || axisToIncrementOn > dims.length) {
        throw new Error('Incorrect axis to increment on');
      }
    }

    for (let k = axisToIncrementOn - 1; k >= 0; --k) {
      index[k]++;
      if (index[k] < dims[k]) {
        break;
      }
      index[k] = 0;
    }
  }

  /**
   * Produces a new dimensions array based on the values in the 'originalDimensions' and 'shape' array
   * Used in Reshape
   * @param originalDims Original Shape array
   * @param shapeHints array containing values to compute the new dimensions
   * For example:
   * originalDims = [2,2] and shapeHints = [0,-1] will return [2,2]
   * originalDims = [2,2] and shapeHints = [4] will return [4]
   * originalDims = [2,2] and shapeHints = [5] will throw an exception
   * https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reshape
   */

  static calculateReshapedDims(originalDims: readonly number[], shapeHints: ArrayLike<number>): number[] {
    // reshape to a Scalar Tensor
    if (shapeHints.length === 0) {
      if (originalDims.length === 0 || ShapeUtil.size(originalDims) === 1) {
        return [];
      } else {
        throw new Error('cannot reshape to a scalar Tensor');
      }
    }

    const nDims = shapeHints.length;
    const reshapedDims = new Array<number>(nDims);
    let unknownDimension = -1;
    let newTensorSize = 1;
    for (let i = 0; i < nDims; i++) {
      if (shapeHints[i] < -1) {
        throw new Error('a dimension in shape hints cannot be less than -1');
      }
      if (shapeHints[i] === -1) {
        if (unknownDimension !== -1) {
          throw new Error('at most one dimension in shape hints can be -1');
        }
        unknownDimension = i;
      } else {
        if (shapeHints[i] === 0) {
          if (i >= originalDims.length) {
            throw new Error('the dimension with value zero exceeds the dimension size of the input tensor');
          }
          reshapedDims[i] = originalDims[i];
        } else {
          reshapedDims[i] = shapeHints[i];
        }
        newTensorSize *= reshapedDims[i];
      }
    }

    const oldTensorSize = ShapeUtil.size(originalDims);
    if (unknownDimension !== -1) {
      if (oldTensorSize % newTensorSize !== 0) {
        throw new Error(`the input tensor cannot be reshaped to the requested shape. Input shape: [${
            originalDims}] Output shape: [${shapeHints}]`);
      }
      reshapedDims[unknownDimension] = oldTensorSize / newTensorSize;
    }
    // validate sizes from originalDims and reshapedDims match
    else {
      if (newTensorSize !== oldTensorSize) {
        throw new Error('reshapedDims and originalDims don\'t have matching sizes');
      }
    }
    return reshapedDims;
  }

  /**
   * Sorts a given array based on the indices in the Perm array
   * Used in Transpose
   * @param a Array to be sorted such as dims or strides
   * @param perm Perm given; if null a will be reversed
   */
  static sortBasedOnPerm(a: readonly number[], perm?: readonly number[]): readonly number[] {
    if (perm) {
      return perm.map((v) => a[v]);
    } else {
      return a.slice().reverse();
    }
  }

  /**
   * Pads a given shape according to the padding values
   * @param dims shape of the Tensor to be padded
   * @param pad pad values
   */
  static padShape(dims: readonly number[], pad: readonly number[]): readonly number[] {
    const rank = dims.length;
    return dims.map((v, i) => v + pad[i] + pad[i + rank]);
  }

  /**
   * Determines if the two shapes are identical
   * @param shape1
   * @param shape2
   */
  static areEqual(shape1: readonly number[], shape2: readonly number[]): boolean {
    if (shape1.length !== shape2.length) {
      return false;
    }
    return shape1.every((v, i) => v === shape2[i]);
  }

  /**
   * Validates if the given `dims` or `shape` is valid in ONNX.js context and returns data size
   * @param dims - input `dims` that needs to be checked
   */
  static validateDimsAndCalcSize(dims: readonly number[]): number {
    if (dims.length > 6) {
      throw new TypeError('Only rank 0 to 6 is supported for tensor shape.');
    }
    let size = 1;
    for (const n of dims) {
      if (!Number.isInteger(n)) {
        throw new TypeError(`Invalid shape: ${n} is not an integer`);
      }
      if (n < 0 || n > 2147483647) {
        throw new TypeError(`Invalid shape: length ${n} is not allowed`);
      }
      size *= n;
    }
    return size;
  }

  /**
   * Determines the shape of output tensor y = flatten(x, axis)
   * @param dims - shape of input tensor
   * @param axis - flatten axis, in the range [-r, r]
   */
  static flattenShape(dims: readonly number[], axis: number): readonly number[] {
    if (axis < 0) {
      axis += dims.length;
    }
    const total = dims.reduce((x, y) => x * y, 1);
    const right = dims.slice(axis).reduce((x, y) => x * y, 1);
    const outputDims = [total / right, right];

    return outputDims;
  }

  /**
   * Determines the shape of output tensor y = squeeze(x, axes)
   * @param dims - shape of input tensor
   * @param axes - squeeze axes
   */
  static squeezeShape(dims: readonly number[], axes: readonly number[]): readonly number[] {
    const outputDims = new Array<number>();

    // sanity check
    axes = ShapeUtil.normalizeAxes(axes, dims.length);

    for (let i = 0; i < dims.length; i++) {
      const inSqueezeList = axes.indexOf(i) >= 0;
      if (inSqueezeList && dims[i] !== 1) {
        throw new Error('squeeze an axis of size different than 1');
      }

      if ((axes.length === 0 && dims[i] > 1) || (axes.length > 0 && !inSqueezeList)) {
        outputDims.push(dims[i]);
      }
    }

    return outputDims;
  }

  /**
   * Determines the shape of output tensor y = unsqueeze(x, axes)
   * @param dims - shape of input tensor
   * @param axes - unsqueeze axes
   */
  static unsqueezeShape(dims: readonly number[], axes: readonly number[]): readonly number[] {
    const outputDims = new Array<number>(dims.length + axes.length);

    // initialize the array elements to 0
    outputDims.fill(0);

    // set all axes indices to 1 in outputDims and check for duplicates
    for (let i = 0; i < axes.length; i++) {
      const axis = ShapeUtil.normalizeAxis(axes[i], outputDims.length);
      if (axis >= outputDims.length) {
        throw new Error('\'axes\' has an out of range axis');
      }
      if (outputDims[axis] !== 0) {
        throw new Error('\'axes\' has a duplicate axis');
      }

      outputDims[axis] = 1;
    }

    // fill in the zero entries of outputDims with the input tensor's shape
    let inputDimsIterator = 0;
    for (let i = 0; i < outputDims.length; i++) {
      if (outputDims[i] === 0) {
        outputDims[i] = dims[inputDimsIterator++];
      }
    }

    // sanity check assertion. 'inputDimsIterator'
    // should be equal to the length of 'dims'
    if (inputDimsIterator !== dims.length) {
      throw new Error('the unsqueezed dimension could not be established');
    }

    return outputDims;
  }
}

export const MIN_CLIP = -3.4028234663852886e+38;
export const MAX_CLIP = 3.4028234663852886e+38;
