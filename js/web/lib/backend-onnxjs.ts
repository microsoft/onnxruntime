// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* eslint-disable import/no-internal-modules */
import {Backend, InferenceSession, SessionHandler} from 'onnxruntime-common';

import {Logger} from './onnxjs/instrument';
import {Session} from './onnxjs/session';
import {OnnxjsSessionHandler} from './onnxjs/session-handler';

class OnnxjsBackend implements Backend {
  protected getOnnxjsSessionConfig(options?: InferenceSession.SessionOptions): Session.Config {
    const sessionConfig: Session.Config = {};
    if (options?.logSeverityLevel) {
      let logLevel: string;
      switch (options.logSeverityLevel as number) {
        case 0:
          logLevel = 'verbose';
          break;
        case 1:
          logLevel = 'info';
          break;
        case 2:
          logLevel = 'warning';
          break;
        case 3:
        case 4:
          logLevel = 'error';
          break;
        default:
          logLevel = 'info';
          break;
      }
      sessionConfig.logger = {};
      sessionConfig.logger.minimalSeverity = logLevel as Logger.Severity;
    }
    return sessionConfig;
  }

  // eslint-disable-next-line @typescript-eslint/no-empty-function
  async init(): Promise<void> {}

  async createSessionHandler(pathOrBuffer: string|Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<SessionHandler> {
    // NOTE: Session.Config(from onnx.js) is not compatible with InferenceSession.SessionOptions(from
    // onnxruntime-common).
    //       In future we should remove Session.Config and use InferenceSession.SessionOptions.
    //       Currently we have to explicitly extract and convert relevant fields.
    const session = new Session(this.getOnnxjsSessionConfig(options));

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
