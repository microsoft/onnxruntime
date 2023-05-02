// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// <reference lib="webworker" />

import {OrtWasmMessage} from '../proxy-messages';
import {createSession, createSessionAllocate, createSessionFinalize, endProfiling, extractTransferableBuffers, initOrt, releaseSession, run} from '../wasm-core-impl';
import {initializeWebAssembly} from '../wasm-factory';

self.onmessage = (ev: MessageEvent<OrtWasmMessage>): void => {
  switch (ev.data.type) {
    case 'init-wasm':
      initializeWebAssembly(ev.data.in)
          .then(
              () => postMessage({type: 'init-wasm'} as OrtWasmMessage),
              err => postMessage({type: 'init-wasm', err} as OrtWasmMessage));
      break;
    case 'init-ort':
      try {
        const {numThreads, loggingLevel} = ev.data.in!;
        initOrt(numThreads, loggingLevel);
        postMessage({type: 'init-ort'} as OrtWasmMessage);
      } catch (err) {
        postMessage({type: 'init-ort', err} as OrtWasmMessage);
      }
      break;
    case 'create_allocate':
      try {
        const {model} = ev.data.in!;
        const modeldata = createSessionAllocate(model);
        postMessage({type: 'create_allocate', out: modeldata} as OrtWasmMessage);
      } catch (err) {
        postMessage({type: 'create_allocate', err} as OrtWasmMessage);
      }
      break;
    case 'create_finalize':
      try {
        const {modeldata, options} = ev.data.in!;
        const sessionMetadata = createSessionFinalize(modeldata, options);
        postMessage({type: 'create_finalize', out: sessionMetadata} as OrtWasmMessage);
      } catch (err) {
        postMessage({type: 'create_finalize', err} as OrtWasmMessage);
      }
      break;
    case 'create':
      try {
        const {model, options} = ev.data.in!;
        const sessionMetadata = createSession(model, options);
        postMessage({type: 'create', out: sessionMetadata} as OrtWasmMessage);
      } catch (err) {
        postMessage({type: 'create', err} as OrtWasmMessage);
      }
      break;
    case 'release':
      try {
        const handler = ev.data.in!;
        releaseSession(handler);
        postMessage({type: 'release'} as OrtWasmMessage);
      } catch (err) {
        postMessage({type: 'release', err} as OrtWasmMessage);
      }
      break;
    case 'run':
      try {
        const {sessionId, inputIndices, inputs, outputIndices, options} = ev.data.in!;
        run(sessionId, inputIndices, inputs, outputIndices, options)
            .then(
                outputs => {
                  postMessage({type: 'run', out: outputs} as OrtWasmMessage, extractTransferableBuffers(outputs));
                },
                err => {
                  postMessage({type: 'run', err} as OrtWasmMessage);
                });
      } catch (err) {
        postMessage({type: 'run', err} as OrtWasmMessage);
      }
      break;
    case 'end-profiling':
      try {
        const handler = ev.data.in!;
        endProfiling(handler);
        postMessage({type: 'end-profiling'} as OrtWasmMessage);
      } catch (err) {
        postMessage({type: 'end-profiling', err} as OrtWasmMessage);
      }
      break;
    default:
  }
};
