// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {env, InferenceSession} from 'onnxruntime-common';

import {OrtWasmMessage, SerializableModeldata, SerializableSessionMetadata, SerializableTensor} from './proxy-messages';
import * as core from './wasm-core-impl';
import {initializeWebAssembly} from './wasm-factory';

const isProxy = (): boolean => !!env.wasm.proxy && typeof document !== 'undefined';
let proxyWorker: Worker|undefined;
let initializing = false;
let initialized = false;
let aborted = false;

// resolve; reject
type PromiseCallbacks<T = void> = [(result: T) => void, (reason: unknown) => void];

let initWasmCallbacks: PromiseCallbacks;
let initOrtCallbacks: PromiseCallbacks;
const createSessionAllocateCallbacks: Array<PromiseCallbacks<SerializableModeldata>> = [];
const createSessionFinalizeCallbacks: Array<PromiseCallbacks<SerializableSessionMetadata>> = [];
const createSessionCallbacks: Array<PromiseCallbacks<SerializableSessionMetadata>> = [];
const releaseSessionCallbacks: Array<PromiseCallbacks<void>> = [];
const runCallbacks: Array<PromiseCallbacks<SerializableTensor[]>> = [];
const endProfilingCallbacks: Array<PromiseCallbacks<void>> = [];

const ensureWorker = (): void => {
  if (initializing || !initialized || aborted || !proxyWorker) {
    throw new Error('worker not ready');
  }
};

const onProxyWorkerMessage = (ev: MessageEvent<OrtWasmMessage>): void => {
  switch (ev.data.type) {
    case 'init-wasm':
      initializing = false;
      if (ev.data.err) {
        aborted = true;
        initWasmCallbacks[1](ev.data.err);
      } else {
        initialized = true;
        initWasmCallbacks[0]();
      }
      break;
    case 'init-ort':
      if (ev.data.err) {
        initOrtCallbacks[1](ev.data.err);
      } else {
        initOrtCallbacks[0]();
      }
      break;
    case 'create_allocate':
      if (ev.data.err) {
        createSessionAllocateCallbacks.shift()![1](ev.data.err);
      } else {
        createSessionAllocateCallbacks.shift()![0](ev.data.out!);
      }
      break;
    case 'create_finalize':
      if (ev.data.err) {
        createSessionFinalizeCallbacks.shift()![1](ev.data.err);
      } else {
        createSessionFinalizeCallbacks.shift()![0](ev.data.out!);
      }
      break;
    case 'create':
      if (ev.data.err) {
        createSessionCallbacks.shift()![1](ev.data.err);
      } else {
        createSessionCallbacks.shift()![0](ev.data.out!);
      }
      break;
    case 'release':
      if (ev.data.err) {
        releaseSessionCallbacks.shift()![1](ev.data.err);
      } else {
        releaseSessionCallbacks.shift()![0]();
      }
      break;
    case 'run':
      if (ev.data.err) {
        runCallbacks.shift()![1](ev.data.err);
      } else {
        runCallbacks.shift()![0](ev.data.out!);
      }
      break;
    case 'end-profiling':
      if (ev.data.err) {
        endProfilingCallbacks.shift()![1](ev.data.err);
      } else {
        endProfilingCallbacks.shift()![0]();
      }
      break;
    default:
  }
};

const scriptSrc = typeof document !== 'undefined' ? (document?.currentScript as HTMLScriptElement)?.src : undefined;

export const initWasm = async(): Promise<void> => {
  if (!BUILD_DEFS.DISABLE_WASM_PROXY && isProxy()) {
    if (initialized) {
      return;
    }
    if (initializing) {
      throw new Error('multiple calls to \'initWasm()\' detected.');
    }
    if (aborted) {
      throw new Error('previous call to \'initWasm()\' failed.');
    }

    initializing = true;

    // overwrite wasm filepaths
    if (env.wasm.wasmPaths === undefined) {
      if (scriptSrc && scriptSrc.indexOf('blob:') !== 0) {
        env.wasm.wasmPaths = scriptSrc.substr(0, +(scriptSrc).lastIndexOf('/') + 1);
      }
    }

    return new Promise<void>((resolve, reject) => {
      proxyWorker?.terminate();
      // eslint-disable-next-line @typescript-eslint/no-var-requires, @typescript-eslint/no-require-imports
      proxyWorker = require('worker-loader?inline=no-fallback!./proxy-worker/main').default() as Worker;
      proxyWorker.onmessage = onProxyWorkerMessage;
      initWasmCallbacks = [resolve, reject];
      const message: OrtWasmMessage = {type: 'init-wasm', in : env.wasm};
      proxyWorker.postMessage(message);
    });

  } else {
    return initializeWebAssembly(env.wasm);
  }
};

export const initOrt = async(numThreads: number, loggingLevel: number): Promise<void> => {
  if (!BUILD_DEFS.DISABLE_WASM_PROXY && isProxy()) {
    ensureWorker();
    return new Promise<void>((resolve, reject) => {
      initOrtCallbacks = [resolve, reject];
      const message: OrtWasmMessage = {type: 'init-ort', in : {numThreads, loggingLevel}};
      proxyWorker!.postMessage(message);
    });
  } else {
    core.initOrt(numThreads, loggingLevel);
  }
};

export const createSessionAllocate = async(model: Uint8Array): Promise<SerializableModeldata> => {
  if (!BUILD_DEFS.DISABLE_WASM_PROXY && isProxy()) {
    ensureWorker();
    return new Promise<SerializableModeldata>((resolve, reject) => {
      createSessionAllocateCallbacks.push([resolve, reject]);
      const message: OrtWasmMessage = {type: 'create_allocate', in : {model}};
      proxyWorker!.postMessage(message, [model.buffer]);
    });
  } else {
    return core.createSessionAllocate(model);
  }
};

export const createSessionFinalize = async(modeldata: SerializableModeldata, options?: InferenceSession.SessionOptions):
    Promise<SerializableSessionMetadata> => {
      if (!BUILD_DEFS.DISABLE_WASM_PROXY && isProxy()) {
        ensureWorker();
        return new Promise<SerializableSessionMetadata>((resolve, reject) => {
          createSessionFinalizeCallbacks.push([resolve, reject]);
          const message: OrtWasmMessage = {type: 'create_finalize', in : {modeldata, options}};
          proxyWorker!.postMessage(message);
        });
      } else {
        return core.createSessionFinalize(modeldata, options);
      }
    };

export const createSession =
    async(model: Uint8Array, options?: InferenceSession.SessionOptions): Promise<SerializableSessionMetadata> => {
  if (!BUILD_DEFS.DISABLE_WASM_PROXY && isProxy()) {
    ensureWorker();
    return new Promise<SerializableSessionMetadata>((resolve, reject) => {
      createSessionCallbacks.push([resolve, reject]);
      const message: OrtWasmMessage = {type: 'create', in : {model, options}};
      proxyWorker!.postMessage(message, [model.buffer]);
    });
  } else {
    return core.createSession(model, options);
  }
};

export const releaseSession = async(sessionId: number): Promise<void> => {
  if (!BUILD_DEFS.DISABLE_WASM_PROXY && isProxy()) {
    ensureWorker();
    return new Promise<void>((resolve, reject) => {
      releaseSessionCallbacks.push([resolve, reject]);
      const message: OrtWasmMessage = {type: 'release', in : sessionId};
      proxyWorker!.postMessage(message);
    });
  } else {
    core.releaseSession(sessionId);
  }
};

export const run = async(
    sessionId: number, inputIndices: number[], inputs: SerializableTensor[], outputIndices: number[],
    options: InferenceSession.RunOptions): Promise<SerializableTensor[]> => {
  if (!BUILD_DEFS.DISABLE_WASM_PROXY && isProxy()) {
    ensureWorker();
    return new Promise<SerializableTensor[]>((resolve, reject) => {
      runCallbacks.push([resolve, reject]);
      const message: OrtWasmMessage = {type: 'run', in : {sessionId, inputIndices, inputs, outputIndices, options}};
      proxyWorker!.postMessage(message, core.extractTransferableBuffers(inputs));
    });
  } else {
    return core.run(sessionId, inputIndices, inputs, outputIndices, options);
  }
};

export const endProfiling = async(sessionId: number): Promise<void> => {
  if (!BUILD_DEFS.DISABLE_WASM_PROXY && isProxy()) {
    ensureWorker();
    return new Promise<void>((resolve, reject) => {
      endProfilingCallbacks.push([resolve, reject]);
      const message: OrtWasmMessage = {type: 'end-profiling', in : sessionId};
      proxyWorker!.postMessage(message);
    });
  } else {
    core.endProfiling(sessionId);
  }
};
