// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { Backend, InferenceSession, InferenceSessionHandler } from 'onnxruntime-common';

import { createDeprecationWarning } from './deprecation-warning';
import { Session } from './onnxjs/session';
import { OnnxjsSessionHandler } from './onnxjs/session-handler-inference';

const warnWebGlDeprecation = createDeprecationWarning(
  'The WebGL execution provider is deprecated and will be removed in a future release. ' +
    "Please migrate to WebGPU ('webgpu') or WebAssembly ('wasm'). " +
    'See https://github.com/microsoft/onnxruntime/issues/32241 for details.',
);

class OnnxjsBackend implements Backend {
  async init(): Promise<void> {
    warnWebGlDeprecation();
  }

  async createInferenceSessionHandler(
    pathOrBuffer: string | Uint8Array,
    options?: InferenceSession.SessionOptions,
  ): Promise<InferenceSessionHandler> {
    // NOTE: Session.Config(from onnx.js) is not compatible with InferenceSession.SessionOptions(from
    // onnxruntime-common).
    //       In future we should remove Session.Config and use InferenceSession.SessionOptions.
    //       Currently we allow this to happen to make test runner work.
    const session = new Session(options as unknown as Session.Config);

    // typescript cannot merge method override correctly (so far in 4.2.3). need if-else to call the method.
    if (typeof pathOrBuffer === 'string') {
      await session.loadModel(pathOrBuffer);
    } else {
      await session.loadModel(pathOrBuffer);
    }

    return new OnnxjsSessionHandler(session);
  }
}

export const onnxjsBackend = new OnnxjsBackend();
