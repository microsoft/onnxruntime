// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Reshape} from '../../../ops/reshape';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {TextureLayout} from '../types';
import {getPackedShape} from '../utils';
import {WebGLReshapePacked} from './reshape-packed';

export class WebGLReshape extends Reshape {
  packedImpl: WebGLReshapePacked;
  constructor() {
    super();
    this.packedImpl = new WebGLReshapePacked();
  }
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    if (inferenceHandler.session.pack) {
      return inferenceHandler.run(this.packedImpl, inputs);
    } else {
      const reshapedDims = ShapeUtil.calculateReshapedDims(inputs[0].dims, inputs[1].integerData);
      const reshapedTensor = reshape(inferenceHandler, inputs[0], reshapedDims);
      return [reshapedTensor];
    }
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

  const newTextureData = inferenceHandler.createSharedTextureData(newTextureLayout, input.type, inputTD.texture);
  return newTextureData.tensor;
}
