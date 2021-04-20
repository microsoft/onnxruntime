// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {SessionHandler} from '../../backend';
import {Graph} from '../../graph';
import {Logger} from '../../instrument';
import {Operator} from '../../operators';
import {OpSet, resolveOperator} from '../../opset';
import {Session} from '../../session';
import {Tensor} from '../../tensor';
import {WebGLBackend} from '../backend-webgl';

import {WebGLInferenceHandler} from './inference-handler';
import {WEBGL_OP_RESOLVE_RULES} from './op-resolve-rules';
import {ProgramManager} from './program-manager';
import {PreferLogicalStrategy, TextureLayoutStrategy} from './texture-layout-strategy';
import {TextureManager} from './texture-manager';
import {TextureData, WebGLOperator} from './types';

export class WebGLSessionHandler implements SessionHandler {
  programManager: ProgramManager;
  textureManager: TextureManager;
  layoutStrategy: TextureLayoutStrategy;
  textureDataCache: Map<Tensor.Id, TextureData>;
  initializers: Set<Tensor.Id>;
  packOpCache: Map<string, WebGLOperator>;
  unpackOpCache: Map<string, WebGLOperator>;

  constructor(public readonly backend: WebGLBackend, public readonly context: Session.Context) {
    this.layoutStrategy = new PreferLogicalStrategy(backend.glContext.maxTextureSize);
    this.programManager = new ProgramManager(this.context.profiler, backend.glContext, this.layoutStrategy);
    this.textureManager = new TextureManager(
        backend.glContext, this.layoutStrategy, this.context.profiler,
        {reuseTextures: backend.textureCacheMode === 'full'});
    this.textureDataCache = new Map();
    this.packOpCache = new Map();
    this.unpackOpCache = new Map();
  }

  createInferenceHandler() {
    return new WebGLInferenceHandler(this);
  }
  onGraphInitialized(graph: Graph): void {
    const initializers = graph.getValues().filter(v => v.from === -1 && v.tensor).map(v => v.tensor!.dataId);
    this.initializers = new Set(initializers);
  }
  isInitializer(tensorId: Tensor.Id): boolean {
    return this.initializers ? this.initializers.has(tensorId) : false;
  }
  getTextureData(tensorId: Tensor.Id): TextureData|undefined {
    return this.textureDataCache.get(tensorId);
  }
  setTextureData(tensorId: Tensor.Id, textureData: TextureData): void {
    Logger.verbose('WebGLSessionHandler', 'Storing Texture data in cache');
    this.textureDataCache.set(tensorId, textureData);
  }
  dispose(): void {
    this.programManager.dispose();
    this.textureManager.clearActiveTextures();
    this.textureDataCache.forEach(td => this.textureManager.releaseTexture(td, true));
    this.textureDataCache = new Map();
  }
  resolve(node: Graph.Node, opsets: readonly OpSet[], graph: Graph): Operator {
    const op = resolveOperator(node, opsets, WEBGL_OP_RESOLVE_RULES);
    op.initialize(node.attributes, node, graph);
    return op;
  }
}
