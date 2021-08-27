// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// <reference lib="webworker" />

import {OrtWasmMessage} from '../proxy-messages';
import {createSession, extractTransferableBuffers, initOrt, releaseSession, run} from '../wasm-core-impl';
import {initializeWebAssembly} from '../wasm-factory';

self.onmessage = (ev: MessageEvent<OrtWasmMessage>): void => {
  switch (ev.data.type) {
    case 'init-wasm':
      initializeWebAssembly(ev.data.in!)
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
        const outputs = run(sessionId, inputIndices, inputs, outputIndices, options);
        postMessage({type: 'run', out: outputs} as OrtWasmMessage, extractTransferableBuffers(outputs));
      } catch (err) {
        postMessage({type: 'run', err} as OrtWasmMessage);
      }
      break;
    default:
  }
};
