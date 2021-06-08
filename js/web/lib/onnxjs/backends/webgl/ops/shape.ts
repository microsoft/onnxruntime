// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Shape} from '../../../ops/shape';
import {Tensor} from '../../../tensor';
import {WebGLInferenceHandler} from '../inference-handler';


export class WebGLShape extends Shape {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return [new Tensor([inputs[0].dims.length], 'int32', undefined, undefined, new Int32Array(inputs[0].dims))];
  }
}
