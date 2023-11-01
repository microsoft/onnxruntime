// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {type Backend, InferenceSession, type InferenceSessionHandler, type SessionHandler, Tensor} from 'onnxruntime-common';
import {Platform} from 'react-native';

import {binding, type Binding, type JSIBlob, jsiHelper} from './binding';

type SupportedTypedArray = Exclude<Tensor.DataType, string[]>;

const tensorTypeToTypedArray = (type: Tensor.Type):|Float32ArrayConstructor|Int8ArrayConstructor|Int16ArrayConstructor|
    Int32ArrayConstructor|BigInt64ArrayConstructor|Float64ArrayConstructor|Uint8ArrayConstructor => {
      switch (type) {
        case 'float32':
          return Float32Array;
        case 'int8':
          return Int8Array;
        case 'uint8':
          return Uint8Array;
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

const normalizePath = (path: string): string => {
  // remove 'file://' prefix in iOS
  if (Platform.OS === 'ios' && path.toLowerCase().startsWith('file://')) {
    return path.substring(7);
  }

  return path;
};

class OnnxruntimeSessionHandler implements InferenceSessionHandler {
  #inferenceSession: Binding.InferenceSession;
  #key: string;

  #pathOrBuffer: string|Uint8Array;

  inputNames: string[];
  outputNames: string[];

  constructor(pathOrBuffer: string|Uint8Array) {
    this.#inferenceSession = binding;
    this.#pathOrBuffer = pathOrBuffer;
    this.#key = '';

    this.inputNames = [];
    this.outputNames = [];
  }

  async loadModel(options: InferenceSession.SessionOptions): Promise<void> {
    try {
      let results: Binding.ModelLoadInfoType;
      // load a model
      if (typeof this.#pathOrBuffer === 'string') {
        // load model from model path
        results = await this.#inferenceSession.loadModel(normalizePath(this.#pathOrBuffer), options);
      } else {
        // load model from buffer
        if (!this.#inferenceSession.loadModelFromBlob) {
          throw new Error('Native module method "loadModelFromBlob" is not defined');
        }
        const modelBlob = jsiHelper.storeArrayBuffer(this.#pathOrBuffer.buffer);
        results = await this.#inferenceSession.loadModelFromBlob(modelBlob, options);
      }
      // resolve promise if onnxruntime session is successfully created
      this.#key = results.key;
      this.inputNames = results.inputNames;
      this.outputNames = results.outputNames;
    } catch (e) {
      throw new Error(`Can't load a model: ${(e as Error).message}`);
    }
  }

  async dispose(): Promise<void> {
    return this.#inferenceSession.dispose(this.#key);
  }

  startProfiling(): void {
    // TODO: implement profiling
  }
  endProfiling(): void {
    // TODO: implement profiling
  }

  async run(feeds: SessionHandler.FeedsType, fetches: SessionHandler.FetchesType, options: InferenceSession.RunOptions):
      Promise<SessionHandler.ReturnType> {
    const outputNames: Binding.FetchesType = [];
    for (const name in fetches) {
      if (Object.prototype.hasOwnProperty.call(fetches, name)) {
        if (fetches[name]) {
          throw new Error(
              'Preallocated output is not supported and only names as string array is allowed as parameter');
        }
        outputNames.push(name);
      }
    }
    const input = this.encodeFeedsType(feeds);
    const results: Binding.ReturnType = await this.#inferenceSession.run(this.#key, input, outputNames, options);
    const output = this.decodeReturnType(results);
    return output;
  }


  encodeFeedsType(feeds: SessionHandler.FeedsType): Binding.FeedsType {
    const returnValue: {[name: string]: Binding.EncodedTensorType} = {};
    for (const key in feeds) {
      if (Object.hasOwnProperty.call(feeds, key)) {
        let data: JSIBlob|string[];

        if (Array.isArray(feeds[key].data)) {
          data = feeds[key].data as string[];
        } else {
          const buffer = (feeds[key].data as SupportedTypedArray).buffer;
          data = jsiHelper.storeArrayBuffer(buffer);
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
          const buffer = jsiHelper.resolveArrayBuffer(results[key].data as JSIBlob) as SupportedTypedArray;
          const typedArray = tensorTypeToTypedArray(results[key].type as Tensor.Type);
          tensorData = new typedArray(buffer, buffer.byteOffset, buffer.byteLength / typedArray.BYTES_PER_ELEMENT);
        }

        returnValue[key] = new Tensor(results[key].type as Tensor.Type, tensorData, results[key].dims);
      }
    }

    return returnValue;
  }
}

class OnnxruntimeBackend implements Backend {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async init(name: string): Promise<void> {
    return Promise.resolve();
  }

  async createInferenceSessionHandler(pathOrBuffer: string|Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<InferenceSessionHandler> {
    const handler = new OnnxruntimeSessionHandler(pathOrBuffer);
    await handler.loadModel(options || {});
    return handler;
  }
}

export const onnxruntimeBackend = new OnnxruntimeBackend();
