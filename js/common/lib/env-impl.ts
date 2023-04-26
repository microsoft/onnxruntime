// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Env} from './env';

type LogLevelType = Env['logLevel'];
export class EnvImpl implements Env {
  constructor() {
    this.wasm = {};
    this.webgl = {};
    this.webgpu = {};
    this.logLevelInternal = 'warning';
  }

  // TODO standadize the getter and setter convention in env for other fields.
  set logLevel(value: LogLevelType) {
    if (value === undefined) {
      return;
    }
    if (typeof value !== 'string' || ['verbose', 'info', 'warning', 'error', 'fatal'].indexOf(value) === -1) {
      throw new Error(`Unsupported logging level: ${value}`);
    }
    this.logLevelInternal = value;
  }
  get logLevel(): LogLevelType {
    return this.logLevelInternal;
  }

  debug?: boolean;

  wasm: Env.WebAssemblyFlags;
  webgl: Env.WebGLFlags;
  webgpu: Env.WebGpuFlags;

  [name: string]: unknown;

  private logLevelInternal: Required<LogLevelType>;
}
