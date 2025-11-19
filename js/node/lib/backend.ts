// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { Backend, InferenceSession, InferenceSessionHandler, SessionHandler } from 'onnxruntime-common';

import { Binding, binding, initOrt } from './binding';

const dataTypeStrings = [
  undefined, // 0
  'float32',
  'uint8',
  'int8',
  'uint16',
  'int16',
  'int32',
  'int64',
  'string',
  'bool',
  'float16',
  'float64',
  'uint32',
  'uint64',
  undefined, // 14
  undefined, // 15
  undefined, // 16
  undefined, // 17
  undefined, // 18
  undefined, // 19
  undefined, // 20
  'uint4',
  'int4',
] as const;

class OnnxruntimeSessionHandler implements InferenceSessionHandler {
  #inferenceSession: Binding.InferenceSession;

  constructor(pathOrBuffer: string | Uint8Array, options: InferenceSession.SessionOptions) {
    initOrt();

    this.#inferenceSession = new binding.InferenceSession();
    if (typeof pathOrBuffer === 'string') {
      this.#inferenceSession.loadModel(pathOrBuffer, options);
    } else {
      this.#inferenceSession.loadModel(pathOrBuffer.buffer, pathOrBuffer.byteOffset, pathOrBuffer.byteLength, options);
    }

    // prepare input/output names and metadata
    this.inputNames = [];
    this.outputNames = [];
    this.inputMetadata = [];
    this.outputMetadata = [];

    // this function takes raw metadata from binding and returns a tuple of the following 2 items:
    // - an array of string representing names
    // - an array of converted InferenceSession.ValueMetadata
    const fillNamesAndMetadata = (
      rawMetadata: readonly Binding.ValueMetadata[],
    ): [names: string[], metadata: InferenceSession.ValueMetadata[]] => {
      const names: string[] = [];
      const metadata: InferenceSession.ValueMetadata[] = [];

      for (const m of rawMetadata) {
        names.push(m.name);
        if (!m.isTensor) {
          metadata.push({ name: m.name, isTensor: false });
        } else {
          const type = dataTypeStrings[m.type];
          if (type === undefined) {
            throw new Error(`Unsupported data type: ${m.type}`);
          }
          const shape: Array<number | string> = [];
          for (let i = 0; i < m.shape.length; ++i) {
            const dim = m.shape[i];
            if (dim === -1) {
              shape.push(m.symbolicDimensions[i]);
            } else if (dim >= 0) {
              shape.push(dim);
            } else {
              throw new Error(`Invalid dimension: ${dim}`);
            }
          }
          metadata.push({
            name: m.name,
            isTensor: m.isTensor,
            type,
            shape,
          });
        }
      }

      return [names, metadata];
    };

    [this.inputNames, this.inputMetadata] = fillNamesAndMetadata(this.#inferenceSession.inputMetadata);
    [this.outputNames, this.outputMetadata] = fillNamesAndMetadata(this.#inferenceSession.outputMetadata);
  }

  async dispose(): Promise<void> {
    this.#inferenceSession.dispose();
  }

  readonly inputNames: string[];
  readonly outputNames: string[];

  readonly inputMetadata: InferenceSession.ValueMetadata[];
  readonly outputMetadata: InferenceSession.ValueMetadata[];

  startProfiling(): void {
    // startProfiling is a no-op.
    //
    // if sessionOptions.enableProfiling is true, profiling will be enabled when the model is loaded.
  }
  endProfiling(): void {
    this.#inferenceSession.endProfiling();
  }

  async run(
    feeds: SessionHandler.FeedsType,
    fetches: SessionHandler.FetchesType,
    options: InferenceSession.RunOptions,
  ): Promise<SessionHandler.ReturnType> {
    return new Promise((resolve, reject) => {
      setImmediate(() => {
        try {
          resolve(this.#inferenceSession.run(feeds, fetches, options));
        } catch (e) {
          // reject if any error is thrown
          reject(e);
        }
      });
    });
  }
}

class OnnxruntimeBackend implements Backend {
  async init(): Promise<void> {
    return Promise.resolve();
  }

  async createInferenceSessionHandler(
    pathOrBuffer: string | Uint8Array,
    options?: InferenceSession.SessionOptions,
  ): Promise<InferenceSessionHandler> {
    return new Promise((resolve, reject) => {
      setImmediate(() => {
        try {
          resolve(new OnnxruntimeSessionHandler(pathOrBuffer, options || {}));
        } catch (e) {
          // reject if any error is thrown
          reject(e);
        }
      });
    });
  }
}

export const onnxruntimeBackend = new OnnxruntimeBackend();
export const listSupportedBackends = binding.listSupportedBackends;
