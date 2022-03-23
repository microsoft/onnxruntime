// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGpuInferenceHandler} from '../inference-handler';

export const reshape = (handler: WebGpuInferenceHandler, inputs: Tensor[]): Tensor[] => {
  const reshapedDims = ShapeUtil.calculateReshapedDims(inputs[0].dims, inputs[1].integerData);
  return [handler.reshape(inputs[0], reshapedDims)];
};
