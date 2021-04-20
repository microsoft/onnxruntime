// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Reshape} from '../../../ops/reshape';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {TextureLayout} from '../types';
import {getPackedShape} from '../utils';

export class WebGLReshape extends Reshape {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    const reshapedDims = ShapeUtil.calculateReshapedDims(inputs[0].dims, inputs[1].integerData);
    const reshapedTensor = reshape(inferenceHandler, inputs[0], reshapedDims);
    return [reshapedTensor];
  }
}

export function reshape(
    inferenceHandler: WebGLInferenceHandler, input: Tensor, reshapedDims: readonly number[]): Tensor {
  const inputTD = inferenceHandler.getOrCreateTextureData(input);
  let packedShape = reshapedDims;
  if (inputTD.channels === 4) {
    packedShape = getPackedShape(reshapedDims);
  }
  const newTextureLayout: TextureLayout = {
    channels: inputTD.channels,
    height: inputTD.height,
    width: inputTD.width,
    // handle reshaping into scalar Tensors
    shape: packedShape.length !== 0 ? packedShape : [1],
    strides: ShapeUtil.computeStrides(packedShape),
    unpackedShape: reshapedDims,
  };

  const newTextureData = inferenceHandler.createSharedTextureData(newTextureLayout, input.type, inputTD.texture, {});
  return newTextureData.tensor;
}
