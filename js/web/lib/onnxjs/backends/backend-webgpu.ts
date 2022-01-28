// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {env} from 'onnxruntime-common';
import {Backend, SessionHandler} from '../backend';
import {Logger} from '../instrument';
import {Session} from '../session';

import {WebGpuSessionHandler} from './webgpu/session-handler';

export class WebGpuBackend implements Backend {
  initialize(): boolean {
    try {
      // STEP.1 TODO: set up context (one time initialization)

      // STEP.2 TODO: set up flags

      Logger.setWithEnv(env);

      Logger.verbose('WebGpuBackend', 'Initialized successfully.');
      return true;
    } catch (e) {
      Logger.warning('WebGpuBackend', `Unable to initialize WebGLBackend. ${e}`);
      return false;
    }
  }
  createSessionHandler(context: Session.Context): SessionHandler {
    return new WebGpuSessionHandler(this, context);
  }
  dispose(): void {
    // TODO: uninitialization
    // this.glContext.dispose();
  }
}
