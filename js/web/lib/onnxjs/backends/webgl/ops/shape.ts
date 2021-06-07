// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Shape} from '../../../ops/shape';
import {Tensor} from '../../../tensor';
import {WebGLInferenceHandler} from '../inference-handler';


export class WebGLShape extends Shape {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    const shape = inputs[0].dims.slice(0);
    return [new Tensor([shape.length], 'int32', undefined, undefined, new Int32Array(shape))];
  }
}
