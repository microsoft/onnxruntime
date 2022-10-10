// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* eslint-disable import/no-internal-modules */
import {Backend, InferenceSession, SessionHandler} from 'onnxruntime-common';

import {Session} from './onnxjs/session';
import {OnnxjsSessionHandler} from './onnxjs/session-handler';

class OnnxjsBackend implements Backend {
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  async init(): Promise<void> {}

  async createSessionHandler(pathOrBuffer: string|Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<SessionHandler> {
    // NOTE: Session.Config(from onnx.js) is not compatible with InferenceSession.SessionOptions(from
    // onnxruntime-common).
    //       In future we should remove Session.Config and use InferenceSession.SessionOptions.
    //       Currently we allow this to happen to make test runner work.
    const onnxjsOptions = {...options as unknown as Session.Config};
    if (!onnxjsOptions.backendHint && options?.executionProviders && options?.executionProviders[0]) {
      const ep = options?.executionProviders[0];
      if (typeof ep === 'string') {
        onnxjsOptions.backendHint = ep;
      } else {
        onnxjsOptions.backendHint = ep.name;
      }
    }
    const session = new Session(onnxjsOptions);

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
