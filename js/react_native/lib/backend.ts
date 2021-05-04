// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Buffer} from 'buffer';
import {Backend, InferenceSession, SessionHandler, Tensor,} from 'onnxruntime-common';

import {binding, Binding} from './binding';

type SupportedTypedArray = Exclude<Tensor.DataType, string[]>;

const tensorTypeToTypedArray = (type: Tensor.Type):|Float32ArrayConstructor|Int8ArrayConstructor|Int16ArrayConstructor|
    Int32ArrayConstructor|BigInt64ArrayConstructor|Float64ArrayConstructor => {
      switch (type) {
        case 'float32':
          return Float32Array;
        case 'int8':
          return Int8Array;
        case 'int16':
          return Int16Array;
        case 'int32':
          return Int32Array;
        case 'bool':
          return Int8Array;
        case 'float64':
          return Float64Array;
        case 'int64':
          /* global BigInt64Array */
          /* eslint no-undef: ["error", { "typeof": true }] */
          return BigInt64Array;
        default:
          throw new Error(`unsupported type: ${type}`);
      }
    };

class OnnxruntimeSessionHandler implements SessionHandler {
  #inferenceSession: Binding.InferenceSession;
  #key: string;

  inputNames: string[];
  outputNames: string[];

  constructor(path: string) {
    this.#inferenceSession = binding;
    this.#key = path;
    this.inputNames = [];
    this.outputNames = [];
  }

  async loadModel(options: InferenceSession.SessionOptions): Promise<void> {
    try {
      // load a model
      const results: Binding.ModelLoadInfoType = await this.#inferenceSession.loadModel(this.#key, options);
      // resolve promise if onnxruntime session is successfully created
      if (results.key !== this.#key) {
        throw new Error('Session key is invalid');
      }

      this.inputNames = results.inputNames;
      this.outputNames = results.outputNames;
      return Promise.resolve();
    } catch (e) {
      throw new Error('Can\'t load a model');
    }
  }

  async dispose(): Promise<void> {
    return Promise.resolve();
  }

  startProfiling(): void {
    // TODO: implement profiling
  }
  endProfiling(): void {
    // TODO: implement profiling
  }

  async run(feeds: SessionHandler.FeedsType, fetches: SessionHandler.FetchesType, options: InferenceSession.RunOptions):
      Promise<SessionHandler.ReturnType> {
    // eslint-disable-next-line no-async-promise-executor
    return new Promise(async (resolve, reject) => {
      try {
        // Java API doesn't support preallocated output names and allows only string array as parameter.
        const outputNames: Binding.FetchesType = [];
        for (const fetch in fetches) {
          if (Object.prototype.hasOwnProperty.call(fetches, fetch)) {
            outputNames.push(fetch);
          }
        }
        const input = this.encodeFeedsType(feeds);
        const results: Binding.ReturnType = await this.#inferenceSession.run(this.#key, input, outputNames, options);
        const output = this.decodeReturnType(results);
        resolve(output);
      } catch (e) {
        reject(e);
      }
    });
  }

  encodeFeedsType(feeds: SessionHandler.FeedsType): Binding.FeedsType {
    const returnValue: {[name: string]: Binding.EncodedTensorType} = {};
    for (const key in feeds) {
      if (Object.hasOwnProperty.call(feeds, key)) {
        let data: string|string[];

        if (Array.isArray(feeds[key].data)) {
          data = feeds[key].data as string[];
        } else {
          // Base64-encode tensor data
          data = Buffer.from((feeds[key].data as SupportedTypedArray).buffer).toString('base64');
        }

        returnValue[key] = {
          dims: feeds[key].dims,
          type: feeds[key].type,
          data,
        };
      }
    }
    return returnValue;
  }

  decodeReturnType(results: Binding.ReturnType): SessionHandler.ReturnType {
    const returnValue: SessionHandler.ReturnType = {};

    for (const key in results) {
      if (Object.hasOwnProperty.call(results, key)) {
        let tensorData: Tensor.DataType;
        if (Array.isArray(results[key].data)) {
          tensorData = results[key].data as string[];
        } else {
          const buffer: Buffer = Buffer.from(results[key].data as string, 'base64');
          const typedArray = tensorTypeToTypedArray(results[key].type as Tensor.Type);
          tensorData = new typedArray(buffer.buffer, buffer.byteOffset, buffer.length / typedArray.BYTES_PER_ELEMENT);
        }

        returnValue[key] = new Tensor(results[key].type as Tensor.Type, tensorData, results[key].dims);
      }
    }

    return returnValue;
  }
}

class OnnxruntimeBackend implements Backend {
  async init(): Promise<void> {
    return Promise.resolve();
  }

  async createSessionHandler(pathOrBuffer: string|Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<SessionHandler> {
    // eslint-disable-next-line no-async-promise-executor
    return new Promise(async (resolve, reject) => {
      try {
        if (typeof pathOrBuffer !== 'string') {
          throw new Error('Uint8Array is not supported');
        }
        const handler = new OnnxruntimeSessionHandler(pathOrBuffer);
        await handler.loadModel(options || {});
        resolve(handler);
      } catch (e) {
        reject(e);
      }
    });
  }
}

export const onnxruntimeBackend = new OnnxruntimeBackend();
