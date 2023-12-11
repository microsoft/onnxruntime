// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// <reference lib="webworker" />

//
// * type hack for "HTMLImageElement"
//
// in typescript, the type of "HTMLImageElement" is defined in lib.dom.d.ts, which is conflict with lib.webworker.d.ts.
// when we use webworker, the lib.webworker.d.ts will be used, which does not have HTMLImageElement defined.
//
// we will get the following errors complaining that HTMLImageElement is not defined:
//
// ====================================================================================================================
//
// ../common/dist/cjs/tensor-factory.d.ts:187:29 - error TS2552: Cannot find name 'HTMLImageElement'. Did you mean
// 'HTMLLIElement'?
//
// 187     fromImage(imageElement: HTMLImageElement, options?: TensorFromImageElementOptions):
// Promise<TypedTensor<'float32'> | TypedTensor<'uint8'>>;
//                                 ~~~~~~~~~~~~~~~~
//
// node_modules/@webgpu/types/dist/index.d.ts:83:7 - error TS2552: Cannot find name 'HTMLImageElement'. Did you mean
// 'HTMLLIElement'?
//
// 83     | HTMLImageElement
//          ~~~~~~~~~~~~~~~~
//
// ====================================================================================================================
//
// `HTMLImageElement` is only used in type declaration and not in real code. So we define it as `unknown` here to
// bypass the type check.
//
declare global {
  type HTMLImageElement = unknown;
}

import {OrtWasmMessage, SerializableTensorMetadata} from '../proxy-messages';
import {createSession, copyFromExternalBuffer, endProfiling, extractTransferableBuffers, initRuntime, releaseSession, run} from '../wasm-core-impl';
import {initializeWebAssembly} from '../wasm-factory';

self.onmessage = (ev: MessageEvent<OrtWasmMessage>): void => {
  switch (ev.data.type) {
    case 'init-wasm':
      try {
        initializeWebAssembly(ev.data.in!)
            .then(
                () => postMessage({type: 'init-wasm'} as OrtWasmMessage),
                err => postMessage({type: 'init-wasm', err} as OrtWasmMessage));
      } catch (err) {
        postMessage({type: 'init-wasm', err} as OrtWasmMessage);
      }
      break;
    case 'init-ort':
      try {
        initRuntime(ev.data.in!).then(() => postMessage({type: 'init-ort'} as OrtWasmMessage), err => postMessage({
                                                                                                 type: 'init-ort',
                                                                                                 err
                                                                                               } as OrtWasmMessage));
      } catch (err) {
        postMessage({type: 'init-ort', err} as OrtWasmMessage);
      }
      break;
    case 'copy_from':
      try {
        const {buffer} = ev.data.in!;
        const bufferData = copyFromExternalBuffer(buffer);
        postMessage({type: 'copy_from', out: bufferData} as OrtWasmMessage);
      } catch (err) {
        postMessage({type: 'copy_from', err} as OrtWasmMessage);
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
        releaseSession(ev.data.in!);
        postMessage({type: 'release'} as OrtWasmMessage);
      } catch (err) {
        postMessage({type: 'release', err} as OrtWasmMessage);
      }
      break;
    case 'run':
      try {
        const {sessionId, inputIndices, inputs, outputIndices, options} = ev.data.in!;
        run(sessionId, inputIndices, inputs, outputIndices, new Array(outputIndices.length).fill(null), options)
            .then(
                outputs => {
                  if (outputs.some(o => o[3] !== 'cpu')) {
                    postMessage({type: 'run', err: 'Proxy does not support non-cpu tensor location.'});
                  } else {
                    postMessage(
                        {type: 'run', out: outputs} as OrtWasmMessage,
                        extractTransferableBuffers(outputs as SerializableTensorMetadata[]));
                  }
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
