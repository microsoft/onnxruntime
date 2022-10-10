// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {SessionHandler} from '../../backend';
import {Graph} from '../../graph';
import {Operator} from '../../operators';
import {OpSet, resolveOperator} from '../../opset';
import {Session} from '../../session';
import {Tensor} from '../../tensor';
import {WebGpuBackend} from '../backend-webgpu';

import {WebGpuInferenceHandler} from './inference-handler';
import {WEBGPU_OP_RESOLVE_RULES} from './op-resolve-rules';
import {ProgramManager} from './program-manager';
import {createTensorDataManager, TensorDataManager} from './tensor-data-manager';

export class WebGpuSessionHandler implements SessionHandler {
  private initializers: Set<Tensor.Id>;
  readonly dataManager: TensorDataManager;
  readonly programManager: ProgramManager;

  constructor(public readonly backend: WebGpuBackend, public readonly context: Session.Context) {
    this.dataManager = createTensorDataManager(this.backend.gpuDataManager);
    this.programManager = new ProgramManager(this.backend, this.context.profiler);
  }

  createInferenceHandler() {
    return new WebGpuInferenceHandler(this);
  }
  onGraphInitialized(graph: Graph): void {
    const initializers = graph.getValues().filter(v => v.from === -1 && v.tensor).map(v => v.tensor!.dataId);
    this.initializers = new Set(initializers);
  }
  isInitializer(tensorId: Tensor.Id): boolean {
    return this.initializers ? this.initializers.has(tensorId) : false;
  }
  addInitializer(tensorId: Tensor.Id): void {
    this.initializers.add(tensorId);
  }
  dispose(): void {
    // TODO
  }
  resolve(node: Graph.Node, opsets: readonly OpSet[], graph: Graph): Operator {
    const op = resolveOperator(node, opsets, WEBGPU_OP_RESOLVE_RULES);
    return {impl: op.opImpl, context: op.opInit ? op.opInit(node, graph) : node};
  }
}
