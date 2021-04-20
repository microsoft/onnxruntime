// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Flatten} from '../../../ops/flatten';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';

import {reshape} from './reshape';

export class WebGLFlatten extends Flatten {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    const outputDims = ShapeUtil.flattenShape(inputs[0].dims, this.axis);

    return [reshape(inferenceHandler, inputs[0], outputDims)];
  }
}
