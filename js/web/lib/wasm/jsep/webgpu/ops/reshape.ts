// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGpuInferenceHandler} from '../inference-handler';

export const reshape = async(handler: WebGpuInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> => {
  validateInputs(inputs);
  const shape = await inputs[1].getData();
  const reshapedDims = ShapeUtil.calculateReshapedDims(inputs[0].dims, shape as Int32Array);
  return [handler.reshape(inputs[0], reshapedDims)];
};

const validateInputs = (inputs: Tensor[]): void => {
  if (!inputs || inputs.length !== 2) {
    throw new Error('Reshape requires 2 inputs.');
  }
  if (inputs[1].type !== 'int32') {
    throw new Error('Invalid input type.');
  }
};
