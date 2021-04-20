// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Squeeze} from '../../../ops/squeeze';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {reshape} from './reshape';

export class WebGLSqueeze extends Squeeze {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    const outputDims = ShapeUtil.squeezeShape(inputs[0].dims, this.axes);
    return [reshape(inferenceHandler, inputs[0], outputDims)];
  }
}
