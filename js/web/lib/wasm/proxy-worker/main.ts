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
import {createSession, copyFromExternalBuffer, endProfiling, extractTransferableBuffers, initEp, initRuntime, releaseSession, run} from '../wasm-core-impl';
import {initializeWebAssembly} from '../wasm-factory';

self.onmessage = (ev: MessageEvent<OrtWasmMessage>): void => {
  const {type, in : message} = ev.data;
  try {
    switch (type) {
      case 'init-wasm':
        initializeWebAssembly(message!.wasm)
            .then(
                () => {
                  initRuntime(message!).then(
                      () => {
                        postMessage({type});
                      },
                      err => {
                        postMessage({type, err});
                      });
                },
                err => {
                  postMessage({type, err});
                });
        break;
      case 'init-ep': {
        const {epName, env} = message!;
        initEp(env, epName)
            .then(
                () => {
                  postMessage({type});
                },
                err => {
                  postMessage({type, err});
                });
        break;
      }
      case 'copy-from': {
        const {buffer} = message!;
        const bufferData = copyFromExternalBuffer(buffer);
        postMessage({type, out: bufferData} as OrtWasmMessage);
        break;
      }
      case 'create': {
        const {model, options} = message!;
        const sessionMetadata = createSession(model, options);
        postMessage({type, out: sessionMetadata} as OrtWasmMessage);
        break;
      }
      case 'release':
        releaseSession(message!);
        postMessage({type});
        break;
      case 'run': {
        const {sessionId, inputIndices, inputs, outputIndices, options} = message!;
        run(sessionId, inputIndices, inputs, outputIndices, new Array(outputIndices.length).fill(null), options)
            .then(
                outputs => {
                  if (outputs.some(o => o[3] !== 'cpu')) {
                    postMessage({type, err: 'Proxy does not support non-cpu tensor location.'});
                  } else {
                    postMessage(
                        {type, out: outputs} as OrtWasmMessage,
                        extractTransferableBuffers(outputs as SerializableTensorMetadata[]));
                  }
                },
                err => {
                  postMessage({type, err});
                });
        break;
      }
      case 'end-profiling':
        endProfiling(message!);
        postMessage({type});
        break;
      default:
    }
  } catch (err) {
    postMessage({type, err} as OrtWasmMessage);
  }
};
