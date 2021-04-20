// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Unsqueeze} from '../../../ops/unsqueeze';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {reshape} from './reshape';

export class WebGLUnsqueeze extends Unsqueeze {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    const outputDims = ShapeUtil.unsqueezeShape(inputs[0].dims, this.axes);
    return [reshape(inferenceHandler, inputs[0], outputDims)];
  }
}
