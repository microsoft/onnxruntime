// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';

import {createPackedReshapeProgramInfo} from './reshape-packed';

export const reshape = (handler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] => {
  if (handler.session.pack) {
    return [handler.run(createPackedReshapeProgramInfo(handler, inputs[0], inputs[1]), [inputs[0]])];
  } else {
    const reshapedDims = ShapeUtil.calculateReshapedDims(inputs[0].dims, inputs[1].integerData);
    const reshapedTensor = handler.reshapeUnpacked(inputs[0], reshapedDims);
    return [reshapedTensor];
  }
};
