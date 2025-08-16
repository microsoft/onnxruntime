// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import type {
  Backend,
  InferenceSession,
  InferenceSessionHandler,
  SessionHandler,
} from 'onnxruntime-common';
import { env, Tensor } from 'onnxruntime-common';
import type { InferenceSessionImpl, ValueMetadata, SessionOptions } from './api';
import { OrtApi, Module } from './binding';

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

type RunOptions = InferenceSession.RunOptions;

const fillNamesAndMetadata = (
  rawMetadata: readonly ValueMetadata[]
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
        const dim = m.shape[i]!;
        if (dim === -1) {
          shape.push(m.symbolicDimensions[i]!);
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

class OnnxruntimeSessionHandler implements InferenceSessionHandler {
  #inferenceSession: InferenceSessionImpl;

  static #initialized = false;

  private constructor(
    session: InferenceSessionImpl,
    info: {
      inputNames: string[];
      outputNames: string[];
      inputMetadata: InferenceSession.ValueMetadata[];
      outputMetadata: InferenceSession.ValueMetadata[];
    }
  ) {
    this.#inferenceSession = session;
    this.inputNames = info.inputNames;
    this.outputNames = info.outputNames;
    this.inputMetadata = info.inputMetadata;
    this.outputMetadata = info.outputMetadata;
  }

  static async create(pathOrBuffer: string | Uint8Array, options: SessionOptions) {
    if (typeof OrtApi === 'undefined') {
      throw new Error(
        'Not found OrtApi, please make sure Onnxruntime installation is successful.'
      );
    }

    if (!OnnxruntimeSessionHandler.#initialized) {
      OnnxruntimeSessionHandler.#initialized = true;

      let logLevel = 2;
      if (env.logLevel) {
        switch (env.logLevel) {
          case 'verbose':
            logLevel = 0;
            break;
          case 'info':
            logLevel = 1;
            break;
          case 'warning':
            logLevel = 2;
            break;
          case 'error':
            logLevel = 3;
            break;
          case 'fatal':
            logLevel = 4;
            break;
          default:
            throw new Error(`Unsupported log level: ${env.logLevel}`);
        }
      }
      OrtApi.initOrtOnce(logLevel, Tensor);
    }

    const session = OrtApi.createInferenceSession();
    if (typeof pathOrBuffer === 'string') {
      await session.loadModel(pathOrBuffer, options);
    } else {
      await session.loadModel(
        pathOrBuffer.buffer as ArrayBuffer,
        pathOrBuffer.byteOffset,
        pathOrBuffer.byteLength,
        options
      );
    }

    const [inputNames, inputMetadata] = fillNamesAndMetadata(
      session.inputMetadata
    );
    const [outputNames, outputMetadata] = fillNamesAndMetadata(
      session.outputMetadata
    );

    return new OnnxruntimeSessionHandler(session, {
      inputNames,
      outputNames,
      inputMetadata,
      outputMetadata,
    });
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
    // if sessionOptions.enableProfiling is true, profiling will be enabled when the model is loaded.
  }

  endProfiling(): void {
    this.#inferenceSession.endProfiling();
  }

  async run(
    feeds: SessionHandler.FeedsType,
    fetches: SessionHandler.FetchesType,
    options: RunOptions
  ): Promise<SessionHandler.ReturnType> {
    return await this.#inferenceSession.run(feeds, fetches, options);
  }
}

class OnnxruntimeBackend implements Backend {
  async init(): Promise<void> {
    return Promise.resolve();
  }

  async createInferenceSessionHandler(
    pathOrBuffer: string | Uint8Array,
    options?: SessionOptions
  ): Promise<InferenceSessionHandler> {
    return await OnnxruntimeSessionHandler.create(pathOrBuffer, {
      ...options,
      ortExtLibPath: Module.ORT_EXTENSIONS_PATH,
    });
  }
}

export const onnxruntimeBackend = new OnnxruntimeBackend();
export const listSupportedBackends = OrtApi.listSupportedBackends;
