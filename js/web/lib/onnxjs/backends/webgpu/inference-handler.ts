// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceHandler} from '../../backend';

import {WebGpuSessionHandler} from './session-handler';

export class WebGpuInferenceHandler implements InferenceHandler {
  constructor(public session: WebGpuSessionHandler) {
    // TODO:
  }

  dispose(): void {}
}
