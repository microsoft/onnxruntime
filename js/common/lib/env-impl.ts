// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Env} from './env';

export class EnvIml implements Env {
  // TODO standadize the getter and setter convention in env for other fields.
  set logLevel(value: 'verbose'|'info'|'warning'|'error'|'fatal'|undefined) {
    if (!value) {
      return;
    }
    if (typeof value !== 'string' || ['verbose', 'info', 'warning', 'error', 'fatal'].indexOf(value) === -1) {
      throw new Error('Unsupported logging level.');
    }
    this.logLevelInternal = value;
  }
  get logLevel(): 'verbose'|'info'|'warning'|'error'|'fatal'|undefined {
    return this.logLevelInternal;
  }

  debug?: boolean;

  wasm: Env.WebAssemblyFlags;

  webgl: Env.WebGLFlags;

  [name: string]: unknown;

  private logLevelInternal?: 'verbose'|'info'|'warning'|'error'|'fatal';
}
