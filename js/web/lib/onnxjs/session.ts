// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {readFile} from 'fs';
import {promisify} from 'util';

import {resolveBackend, SessionHandlerType} from './backend';
import {ExecutionPlan} from './execution-plan';
import {Graph} from './graph';
import {Profiler} from './instrument';
import {Model} from './model';
import {Operator} from './operators';
import {Tensor} from './tensor';

export declare namespace Session {
  export interface Config {
    backendHint?: string;
    profiler?: Profiler.Config;
  }

  export interface Context {
    profiler: Readonly<Profiler>;
    graphInputTypes?: Tensor.DataType[];
    graphInputDims?: Array<readonly number[]>;
  }
}

export class Session {
  constructor(config: Session.Config = {}) {
    this._initialized = false;
    this.backendHint = config.backendHint;
    this.profiler = Profiler.create(config.profiler);
    this.context = {profiler: this.profiler, graphInputTypes: [], graphInputDims: []};
  }

  get inputNames(): readonly string[] {
    return this._model.graph.getInputNames();
  }
  get outputNames(): readonly string[] {
    return this._model.graph.getOutputNames();
  }

  startProfiling() {
    this.profiler.start();
  }

  endProfiling() {
    this.profiler.stop();
  }

  async loadModel(uri: string): Promise<void>;
  async loadModel(buffer: ArrayBuffer, byteOffset?: number, length?: number): Promise<void>;
  async loadModel(buffer: Uint8Array): Promise<void>;
  async loadModel(arg: string|ArrayBuffer|Uint8Array, byteOffset?: number, length?: number): Promise<void> {
    await this.profiler.event('session', 'Session.loadModel', async () => {
      // resolve backend and session handler
      const backend = await resolveBackend(this.backendHint);
      this.sessionHandler = backend.createSessionHandler(this.context);

      this._model = new Model();
      if (typeof arg === 'string') {
        const isOrtFormat = arg.endsWith('.ort');
        if (typeof fetch === 'undefined') {
          // node
          const buf = await promisify(readFile)(arg);
          this.initialize(buf, isOrtFormat);
        } else {
          // browser
          const response = await fetch(arg);
          const buf = await response.arrayBuffer();
          this.initialize(new Uint8Array(buf), isOrtFormat);
        }
      } else if (!ArrayBuffer.isView(arg)) {
        // load model from ArrayBuffer
        const arr = new Uint8Array(arg, byteOffset || 0, length || arg.byteLength);
        this.initialize(arr);
      } else {
        // load model from Uint8array
        this.initialize(arg);
      }
    });
  }

  private initialize(modelProtoBlob: Uint8Array, isOrtFormat?: boolean): void {
    if (this._initialized) {
      throw new Error('already initialized');
    }

    this.profiler.event('session', 'Session.initialize', () => {
      // load graph
      const graphInitializer =
          this.sessionHandler.transformGraph ? this.sessionHandler as Graph.Initializer : undefined;
      this._model.load(modelProtoBlob, graphInitializer, isOrtFormat);

      // graph is completely initialzied at this stage , let the interested handlers know
      if (this.sessionHandler.onGraphInitialized) {
        this.sessionHandler.onGraphInitialized(this._model.graph);
      }
      // initialize each operator in the graph
      this.initializeOps(this._model.graph);

      // instantiate an ExecutionPlan object to be used by the Session object
      this._executionPlan = new ExecutionPlan(this._model.graph, this._ops, this.profiler);
    });

    this._initialized = true;
  }

  async run(inputs: Map<string, Tensor>|Tensor[]): Promise<Map<string, Tensor>> {
    if (!this._initialized) {
      throw new Error('session not initialized yet');
    }

    return this.profiler.event('session', 'Session.run', async () => {
      const inputTensors = this.normalizeAndValidateInputs(inputs);

      const outputTensors = await this._executionPlan.execute(this.sessionHandler, inputTensors);

      return this.createOutput(outputTensors);
    });
  }

  private normalizeAndValidateInputs(inputs: Map<string, Tensor>|Tensor[]): Tensor[] {
    const modelInputNames = this._model.graph.getInputNames();

    // normalize inputs
    // inputs: Tensor[]
    if (Array.isArray(inputs)) {
      if (inputs.length !== modelInputNames.length) {
        throw new Error(`incorrect input array length: expected ${modelInputNames.length} but got ${inputs.length}`);
      }
    }
    // convert map to array
    // inputs: Map<string, Tensor>
    else {
      if (inputs.size !== modelInputNames.length) {
        throw new Error(`incorrect input map size: expected ${modelInputNames.length} but got ${inputs.size}`);
      }

      const sortedInputs = new Array<Tensor>(inputs.size);
      let sortedInputsIndex = 0;
      for (let i = 0; i < modelInputNames.length; ++i) {
        const tensor = inputs.get(modelInputNames[i]);
        if (!tensor) {
          throw new Error(`missing input tensor for: '${name}'`);
        }
        sortedInputs[sortedInputsIndex++] = tensor;
      }

      inputs = sortedInputs;
    }

    // validate dims requirements
    // First session run - graph input data is not cached for the session
    if (!this.context.graphInputTypes || this.context.graphInputTypes.length === 0 || !this.context.graphInputDims ||
        this.context.graphInputDims.length === 0) {
      const modelInputIndices = this._model.graph.getInputIndices();
      const modelValues = this._model.graph.getValues();

      const graphInputDims = new Array<readonly number[]>(modelInputIndices.length);

      for (let i = 0; i < modelInputIndices.length; ++i) {
        const graphInput = modelValues[modelInputIndices[i]];
        graphInputDims[i] = graphInput.type!.shape.dims;

        // cached for second and subsequent runs.
        // Some parts of the framework works on the assumption that the graph and types and shapes are static
        this.context.graphInputTypes!.push(graphInput.type!.tensorType);
        this.context.graphInputDims!.push(inputs[i].dims);
      }

      this.validateInputTensorDims(graphInputDims, inputs, true);
    }

    // Second and subsequent session runs - graph input data is cached for the session
    else {
      this.validateInputTensorDims(this.context.graphInputDims, inputs, false);
    }

    // validate types requirement
    this.validateInputTensorTypes(this.context.graphInputTypes!, inputs);

    return inputs;
  }

  private validateInputTensorTypes(graphInputTypes: Tensor.DataType[], givenInputs: Tensor[]) {
    for (let i = 0; i < givenInputs.length; i++) {
      const expectedType = graphInputTypes[i];
      const actualType = givenInputs[i].type;
      if (expectedType !== actualType) {
        throw new Error(`input tensor[${i}] check failed: expected type '${expectedType}' but got ${actualType}`);
      }
    }
  }

  private validateInputTensorDims(
      graphInputDims: Array<readonly number[]>, givenInputs: Tensor[], noneDimSupported: boolean) {
    for (let i = 0; i < givenInputs.length; i++) {
      const expectedDims = graphInputDims[i];
      const actualDims = givenInputs[i].dims;
      if (!this.compareTensorDims(expectedDims, actualDims, noneDimSupported)) {
        throw new Error(`input tensor[${i}] check failed: expected shape '[${expectedDims.join(',')}]' but got [${
            actualDims.join(',')}]`);
      }
    }
  }

  private compareTensorDims(expectedDims: readonly number[], actualDims: readonly number[], noneDimSupported: boolean):
      boolean {
    if (expectedDims.length !== actualDims.length) {
      return false;
    }

    for (let i = 0; i < expectedDims.length; ++i) {
      if (expectedDims[i] !== actualDims[i] && (!noneDimSupported || expectedDims[i] !== 0)) {
        // data shape mis-match AND not a 'None' dimension.
        return false;
      }
    }

    return true;
  }

  private createOutput(outputTensors: Tensor[]): Map<string, Tensor> {
    const modelOutputNames = this._model.graph.getOutputNames();
    if (outputTensors.length !== modelOutputNames.length) {
      throw new Error('expected number of outputs do not match number of generated outputs');
    }

    const output = new Map<string, Tensor>();
    for (let i = 0; i < modelOutputNames.length; ++i) {
      output.set(modelOutputNames[i], outputTensors[i]);
    }

    return output;
  }

  private initializeOps(graph: Graph): void {
    const nodes = graph.getNodes();
    this._ops = new Array(nodes.length);

    for (let i = 0; i < nodes.length; i++) {
      this._ops[i] = this.sessionHandler.resolve(nodes[i], this._model.opsets, graph);
    }
  }

  private _model: Model;
  private _initialized: boolean;

  private _ops: Operator[];
  private _executionPlan: ExecutionPlan;

  private backendHint?: string;

  private sessionHandler: SessionHandlerType;
  private context: Session.Context;
  private profiler: Readonly<Profiler>;
}
