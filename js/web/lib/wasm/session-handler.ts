// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {readFile} from 'fs';
import {env, InferenceSession, SessionHandler, Tensor} from 'onnxruntime-common';
import {promisify} from 'util';

import {SerializableModeldata} from './proxy-messages';
import {createSession, createSessionAllocate, createSessionFinalize, endProfiling, initializeRuntime, releaseSession, run} from './proxy-wrapper';

let runtimeInitialized: boolean;

export class OnnxruntimeWebAssemblySessionHandler implements SessionHandler {
  private sessionId: number;

  inputNames: string[];
  outputNames: string[];

  async createSessionAllocate(path: string): Promise<SerializableModeldata> {
    // fetch model from url and move to wasm heap. The arraybufffer that held the http
    // response is freed once we return
    const response = await fetch(path);
    const arrayBuffer = await response.arrayBuffer();
    return createSessionAllocate(new Uint8Array(arrayBuffer));
  }

  async loadModel(pathOrBuffer: string|Uint8Array, options?: InferenceSession.SessionOptions): Promise<void> {
    if (!runtimeInitialized) {
      await initializeRuntime(env);
      runtimeInitialized = true;
    }

    if (typeof pathOrBuffer === 'string') {
      if (typeof process !== 'undefined' && process.versions && process.versions.node) {
        // node
        const model = await promisify(readFile)(pathOrBuffer);
        [this.sessionId, this.inputNames, this.outputNames] = await createSession(model, options);
      } else {
        // browser
        // fetch model and move to wasm heap.
        const modelData: SerializableModeldata = await this.createSessionAllocate(pathOrBuffer);
        // create the session
        [this.sessionId, this.inputNames, this.outputNames] = await createSessionFinalize(modelData, options);
      }
    } else {
      [this.sessionId, this.inputNames, this.outputNames] = await createSession(pathOrBuffer, options);
    }
  }

  async dispose(): Promise<void> {
    return releaseSession(this.sessionId);
  }

  async run(feeds: SessionHandler.FeedsType, fetches: SessionHandler.FetchesType, options: InferenceSession.RunOptions):
      Promise<SessionHandler.ReturnType> {
    const inputArray: Tensor[] = [];
    const inputIndices: number[] = [];
    Object.entries(feeds).forEach(kvp => {
      const name = kvp[0];
      const tensor = kvp[1];
      const index = this.inputNames.indexOf(name);
      if (index === -1) {
        throw new Error(`invalid input '${name}'`);
      }
      inputArray.push(tensor);
      inputIndices.push(index);
    });

    const outputIndices: number[] = [];
    Object.entries(fetches).forEach(kvp => {
      const name = kvp[0];
      // TODO: support pre-allocated output
      const index = this.outputNames.indexOf(name);
      if (index === -1) {
        throw new Error(`invalid output '${name}'`);
      }
      outputIndices.push(index);
    });

    const outputs =
        await run(this.sessionId, inputIndices, inputArray.map(t => [t.type, t.dims, t.data]), outputIndices, options);

    const result: SessionHandler.ReturnType = {};
    for (let i = 0; i < outputs.length; i++) {
      result[this.outputNames[outputIndices[i]]] = new Tensor(outputs[i][0], outputs[i][2], outputs[i][1]);
    }
    return result;
  }

  startProfiling(): void {
    // TODO: implement profiling
  }

  endProfiling(): void {
    void endProfiling(this.sessionId);
  }
}
