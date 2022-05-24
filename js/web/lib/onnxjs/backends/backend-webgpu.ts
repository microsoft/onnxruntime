// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {env} from 'onnxruntime-common';

import {Backend, SessionHandler} from '../backend';
import {Logger} from '../instrument';
import {Session} from '../session';

import {WebGpuSessionHandler} from './webgpu/session-handler';

export class WebGpuBackend implements Backend {
  device: GPUDevice;
  async initialize(): Promise<boolean> {
    try {
      if (!navigator.gpu) {
        // WebGPU is not available.
        Logger.warning('WebGpuBackend', 'WebGPU is not available.');
        return false;
      }

      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        Logger.warning('WebGpuBackend', 'Failed to get GPU adapter.');
        return false;
      }
      this.device = await adapter.requestDevice();

      // TODO: set up flags

      Logger.setWithEnv(env);

      Logger.verbose('WebGpuBackend', 'Initialized successfully.');

      this.device.onuncapturederror = ev => {
        if (ev.error instanceof GPUValidationError) {
          // eslint-disable-next-line no-console
          console.error(`An uncaught WebGPU validation error was raised: ${ev.error.message}`);
        }
      };

      return true;
    } catch (e) {
      Logger.warning('WebGpuBackend', `Unable to initialize WebGpuBackend. ${e}`);
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
