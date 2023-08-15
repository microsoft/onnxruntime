// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {readFile} from 'fs';
import {env, InferenceSession, SessionHandler, Tensor} from 'onnxruntime-common';
import {promisify} from 'util';

import {SerializableModeldata} from './proxy-messages';
import {createSessionAllocate, createSessionFinalize, endProfiling, initializeRuntime, releaseSession, run} from './proxy-wrapper';
import { streamResponseToBuffer } from './wasm-common';

let runtimeInitialized: boolean;

export class OnnxruntimeWebAssemblySessionHandler implements SessionHandler {
  private sessionId: number;

  inputNames: string[];
  outputNames: string[];

  async fetchModelAndWeights(modelPath: string, weightsPath?: string): Promise<[Uint8Array, ArrayBuffer?]> {
    const modelResponse = await fetch(modelPath);
    const promises: [Promise<Uint8Array>, Promise<ArrayBuffer>?] = [
      modelResponse.arrayBuffer().then(b => new Uint8Array(b))
    ];

    if (weightsPath) {
      const weightsResponse = await fetch(weightsPath);
      const weightsSize = parseInt(weightsResponse.headers.get('Content-Length')!, 10);
      // we cannot create ArrayBuffer > 2gb but 64bit WASM Memory can have arbitrary size
      const weightsMemory = new WebAssembly.Memory({
        initial: Math.ceil(weightsSize / 65536),
        maximum: Math.ceil(weightsSize / 65536),
        // WASM Memory "index" parameter spec change landed but types are not yet updated
        // https://github.com/WebAssembly/memory64/pull/39
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        index: 'u64',
        shared: true,
      });
      promises.push(streamResponseToBuffer(weightsResponse, weightsMemory.buffer, 0)
        .then(() => weightsMemory.buffer));
    }

    // fetch model and weights in parallel
    return Promise.all(promises);
  }

  async loadModel(urisOrBuffers: string|[string, string]|Uint8Array|[Uint8Array, ArrayBuffer],
    options?: InferenceSession.SessionOptions): Promise<void> {
    if (!runtimeInitialized) {
      await initializeRuntime(env);
      runtimeInitialized = true;
    }

    let modelBuffer: Uint8Array;
    let weightsBuffer: ArrayBuffer|undefined;
    const isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
    if (Array.isArray(urisOrBuffers)) {
      // handle [string, string]
      if (typeof urisOrBuffers[0] === 'string') {
        if (isNode) {
          modelBuffer = await promisify(readFile)(urisOrBuffers[0]);
          weightsBuffer = await promisify(readFile)(urisOrBuffers[1] as string);
        } else {
          [modelBuffer, weightsBuffer] = await this.fetchModelAndWeights(urisOrBuffers[0], urisOrBuffers[1] as string);
        }
      } else { // [UInt8Array, ArrayBuffer]
        [modelBuffer, weightsBuffer] = urisOrBuffers as [Uint8Array, ArrayBuffer];
      }
    } else {
      if (typeof urisOrBuffers === 'string') {
        if (isNode) {
          modelBuffer = await promisify(readFile)(urisOrBuffers);
        } else {
          [modelBuffer] = await this.fetchModelAndWeights(urisOrBuffers);
        }
      } else {
        modelBuffer = urisOrBuffers;
      }
    }

    const modelData: SerializableModeldata = await createSessionAllocate(modelBuffer, weightsBuffer);
    // create the session
    [this.sessionId, this.inputNames, this.outputNames] = await createSessionFinalize(modelData, options);
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
      result[this.outputNames[outputIndices[i]]] = new Tensor(
          outputs[i][0], outputs[i][2], outputs[i][1].map(i => Number(i))
      );
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
