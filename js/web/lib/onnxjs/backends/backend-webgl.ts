// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {flags} from '../../backend-onnxjs';
import {Backend, SessionHandler} from '../backend';
import {Logger} from '../instrument';
import {Session} from '../session';

import {WebGLSessionHandler} from './webgl/session-handler';
import {WebGLContext} from './webgl/webgl-context';
import {createWebGLContext} from './webgl/webgl-context-factory';

/**
 * WebGLBackend is the entry point for all WebGL opeartions
 * When it starts it created the WebGLRenderingContext
 * and other main framework components such as Program and Texture Managers
 */
export class WebGLBackend implements Backend {
  glContext: WebGLContext;

  get contextId(): 'webgl'|'webgl2'|undefined {
    return flags.contextId;
  }
  set contextId(value: 'webgl'|'webgl2'|undefined) {
    flags.contextId = value;
  }

  get matmulMaxBatchSize(): number|undefined {
    return flags.matmulMaxBatchSize;
  }
  set matmulMaxBatchSize(value: number|undefined) {
    flags.matmulMaxBatchSize = value;
  }

  get textureCacheMode(): 'initializerOnly'|'full'|undefined {
    return flags.textureCacheMode;
  }
  set textureCacheMode(value: 'initializerOnly'|'full'|undefined) {
    flags.textureCacheMode = value;
  }

  get pack(): boolean|undefined {
    return flags.pack;
  }
  set pack(value: boolean|undefined) {
    flags.pack = value;
  }

  initialize(): boolean {
    try {
      this.glContext = createWebGLContext(this.contextId);
      if (typeof this.matmulMaxBatchSize !== 'number') {
        this.matmulMaxBatchSize = 16;
      }
      if (typeof this.textureCacheMode !== 'string') {
        this.textureCacheMode = 'full';
      }
      if (typeof this.pack !== 'boolean') {
        this.pack = false;
      }
      Logger.verbose(
          'WebGLBackend',
          `Created WebGLContext: ${typeof this.glContext} with matmulMaxBatchSize: ${
              this.matmulMaxBatchSize}; textureCacheMode: ${this.textureCacheMode}; pack: ${this.pack}.`);
      return true;
    } catch (e) {
      Logger.warning('WebGLBackend', `Unable to initialize WebGLBackend. ${e}`);
      return false;
    }
  }
  createSessionHandler(context: Session.Context): SessionHandler {
    return new WebGLSessionHandler(this, context);
  }
  dispose(): void {
    this.glContext.dispose();
  }
}
