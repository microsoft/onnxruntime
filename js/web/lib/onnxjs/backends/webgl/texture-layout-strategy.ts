// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Logger} from '../../instrument';
import {assert} from '../../util';

/** Layout preferences */
export interface WidthHeightPrefs {
  breakAxis?: number;
  isPacked?: boolean;
  reverseWH?: boolean;
}
/**
 * TextureLayoutStrategy is an abstraction for different plans
 * for mapping n-dimensional arrays to 2D textures (and back)
 */
export interface TextureLayoutStrategy {
  computeTextureWH(shape: readonly number[], prefs?: WidthHeightPrefs): [number, number];
}

/**
 * This strategy try to find the minimal max(W,H) that fulfills (W * H == totalSize)
 */
export class AlwaysKeepOriginalSizeStrategy implements TextureLayoutStrategy {
  constructor(public maxTextureSize: number) {}
  computeTextureWH(shape: readonly number[], prefs?: WidthHeightPrefs): [number, number] {
    // scalar tensor
    if (shape.length === 0) {
      return [1, 1];
    }
    const maxTextureSize = this.maxTextureSize;
    if (prefs && prefs.breakAxis !== undefined) {
      // check to see if dims fit
      const wsize = prefs.breakAxis >= shape.length ? 1 : shape.slice(prefs.breakAxis).reduce((a, b) => a * b);
      const hsize = prefs.breakAxis <= 0 ? 1 : shape.slice(0, prefs.breakAxis).reduce((a, b) => a * b);
      if (wsize > maxTextureSize || hsize > maxTextureSize) {
        // ignore preferences
        // continue with default layout
        Logger.verbose(
            'TextureLayout',
            `Given width/height preferences were unattainable: shape:${shape}, breakAxis:${prefs.breakAxis}`);
      } else {
        return [wsize, hsize];
      }
    }
    const totalSize = shape.reduce((a, b) => a * b);

    let width = Math.floor(Math.sqrt(totalSize));

    for (; width < maxTextureSize && width < totalSize; width++) {
      if (totalSize % width === 0) {
        break;
      }
    }

    if (width >= maxTextureSize || totalSize % width !== 0) {
      throw new Error(`The given dimensions are outside this GPU's boundaries: ${shape}`);
    }
    return [width, totalSize / width];
  }
}

export class PreferLogicalStrategy implements TextureLayoutStrategy {
  constructor(public maxTextureSize: number) {}
  computeTextureWH(shape: readonly number[], prefs?: WidthHeightPrefs): [number, number] {
    const wh = this.computeTexture(shape, prefs);
    if (prefs && prefs.isPacked) {
      wh[0] /= 2;
      wh[1] /= 2;
    }
    if (prefs && prefs.reverseWH) {
      return [wh[1], wh[0]];
    }
    return wh;
  }

  computeTexture(shape: readonly number[], prefs?: WidthHeightPrefs): [number, number] {
    const isPacked = prefs && prefs.isPacked;
    // scalar tensor
    if (shape.length === 0) {
      return isPacked ? [2, 2] : [1, 1];
    }
    let maxTextureSize = this.maxTextureSize;
    if (prefs && prefs.breakAxis !== undefined) {
      // check to see if dims fit
      const wsize = prefs.breakAxis >= shape.length ? 1 : shape.slice(prefs.breakAxis).reduce((a, b) => a * b);
      const hsize = prefs.breakAxis <= 0 ? 1 : shape.slice(0, prefs.breakAxis).reduce((a, b) => a * b);
      if (wsize > maxTextureSize || hsize > maxTextureSize) {
        // ignore preferences
        // continue with default layout
        Logger.verbose(
            'TextureLayout',
            `Given width/height preferences were unattainable: shape:${shape}, breakAxis:${prefs.breakAxis}`);
      } else {
        return [wsize, hsize];
      }
    }
    let logShape = shape.slice(0);
    if (isPacked) {
      maxTextureSize = maxTextureSize * 2;

      // This logic ensures we accurately count the number of packed texels needed
      // to accommodate the tensor. We can only pack values in the same texel if
      // they are from adjacent pairs of rows/cols within the same batch. So if a
      // tensor has 3 rows, we pretend it has 4 rows in order to account for the
      // fact that the texels containing the third row are half empty.
      logShape = logShape.map(
          (d, i) => i >= logShape.length - 2 ? (logShape[i] % 2 === 0 ? logShape[i] : logShape[i] + 1) : logShape[i]);

      // Packed texture height is at least 2 (the channel height of a single
      // texel).
      if (logShape.length === 1) {
        logShape = [2, logShape[0]];
      }
    }

    // If logical shape is 2, we don't squeeze, since we want to match physical.
    if (logShape.length !== 2) {
      const squeezeResult = squeezeShape(logShape);
      logShape = squeezeResult.newShape;
    }

    const size = sizeFromShape(logShape);
    if (logShape.length <= 1 && size <= maxTextureSize) {
      return [1, size];
    } else if (logShape.length === 2 && logShape[0] <= maxTextureSize && logShape[1] <= maxTextureSize) {
      return logShape as [number, number];
    } else if (logShape.length === 3 && logShape[0] * logShape[1] <= maxTextureSize && logShape[2] <= maxTextureSize) {
      return [logShape[0] * logShape[1], logShape[2]];
    } else if (logShape.length === 3 && logShape[0] <= maxTextureSize && logShape[1] * logShape[2] <= maxTextureSize) {
      return [logShape[0], logShape[1] * logShape[2]];
    } else if (
        logShape.length === 4 && logShape[0] * logShape[1] * logShape[2] <= maxTextureSize &&
        logShape[3] <= maxTextureSize) {
      return [logShape[0] * logShape[1] * logShape[2], logShape[3]];
    } else if (
        logShape.length === 4 && logShape[0] <= maxTextureSize &&
        logShape[1] * logShape[2] * logShape[3] <= maxTextureSize) {
      return [logShape[0], logShape[1] * logShape[2] * logShape[3]];
    } else {
      if (isPacked) {
        // For packed textures size equals the number of channels required to
        // accommodate the texture data. However in order to squarify such that
        // inner dimensions stay even, we rewrite size to equal the number of
        // texels. Then in the return statement we rehydrate the squarified
        // dimensions to channel units.
        return sizeToSquarishShape(size / 4).map(d => d * 2) as [number, number];
      }
      return sizeToSquarishShape(size);
    }
  }
}

export function squeezeShape(shape: number[], axis?: number[]): {newShape: number[]; keptDims: number[]} {
  const newShape: number[] = [];
  const keptDims: number[] = [];
  const isEmptyArray = axis != null && Array.isArray(axis) && axis.length === 0;
  const axes = (axis == null || isEmptyArray) ? null : parseAxisParam(axis, shape).sort();
  let j = 0;
  for (let i = 0; i < shape.length; ++i) {
    if (axes != null) {
      if (axes[j] === i && shape[i] !== 1) {
        throw new Error(`Can't squeeze axis ${i} since its dim '${shape[i]}' is not 1`);
      }
      if ((axes[j] == null || axes[j] > i) && shape[i] === 1) {
        newShape.push(shape[i]);
        keptDims.push(i);
      }
      if (axes[j] <= i) {
        j++;
      }
    }
    if (shape[i] !== 1) {
      newShape.push(shape[i]);
      keptDims.push(i);
    }
  }
  return {newShape, keptDims};
}

export function parseAxisParam(axis: number|number[], shape: number[]): number[] {
  const rank = shape.length;

  // Normalize input
  axis = axis == null ? shape.map((s, i) => i) : ([] as number[]).concat(axis);

  // Check for valid range
  assert(
      axis.every(ax => ax >= -rank && ax < rank),
      () => `All values in axis param must be in range [-${rank}, ${rank}) but ` +
          `got axis ${axis}`);

  // Check for only integers
  assert(
      axis.every(isInt),
      () => 'All values in axis param must be integers but ' +
          `got axis ${axis}`);

  // Handle negative axis.
  return axis.map(a => a < 0 ? rank + a : a);
}
export function isInt(a: number): boolean {
  return a % 1 === 0;
}
export function sizeFromShape(shape: number[]): number {
  if (shape.length === 0) {
    // Scalar.
    return 1;
  }
  let size = shape[0];
  for (let i = 1; i < shape.length; i++) {
    size *= shape[i];
  }
  return size;
}
export function getRowsCols(shape: number[]): [number, number] {
  if (shape.length === 0) {
    throw Error('Cannot get rows and columns of an empty shape array.');
  }

  return [shape.length > 1 ? shape[shape.length - 2] : 1, shape[shape.length - 1]];
}
export function sizeToSquarishShape(size: number): [number, number] {
  const width = Math.ceil(Math.sqrt(size));
  return [width, Math.ceil(size / width)];
}
export function getBatchDim(shape: number[], dimsToSkip = 2): number {
  return sizeFromShape(shape.slice(0, shape.length - dimsToSkip));
}
