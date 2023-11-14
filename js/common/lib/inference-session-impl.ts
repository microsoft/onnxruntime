// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {resolveBackend} from './backend-impl.js';
import {InferenceSessionHandler} from './backend.js';
import type {InferenceSession as InferenceSessionInterface, MaybePromise} from './inference-session.js';
import {OnnxValue} from './onnx-value.js';
import {Tensor} from './tensor.js';

type SessionOptions = InferenceSessionInterface.SessionOptions;
type RunOptions = InferenceSessionInterface.RunOptions;
type FeedsType = InferenceSessionInterface.FeedsType;
type FetchesType = InferenceSessionInterface.FetchesType;
type ReturnType = InferenceSessionInterface.ReturnType;

export class InferenceSession implements InferenceSessionInterface {
  private constructor(handler: InferenceSessionHandler) {
    this.handler = handler;
  }
  run(feeds: FeedsType, options?: RunOptions): Promise<ReturnType>;
  run(feeds: FeedsType, fetches: FetchesType, options?: RunOptions): Promise<ReturnType>;
  async run(feeds: FeedsType, arg1?: FetchesType|RunOptions, arg2?: RunOptions): Promise<ReturnType> {
    const fetches: {[name: string]: OnnxValue|null} = {};
    let options: RunOptions = {};
    // check inputs
    if (typeof feeds !== 'object' || feeds === null || feeds instanceof Tensor || Array.isArray(feeds)) {
      throw new TypeError(
          '\'feeds\' must be an object that use input names as keys and OnnxValue as corresponding values.');
    }

    let isFetchesEmpty = true;
    // determine which override is being used
    if (typeof arg1 === 'object') {
      if (arg1 === null) {
        throw new TypeError('Unexpected argument[1]: cannot be null.');
      }
      if (arg1 instanceof Tensor) {
        throw new TypeError('\'fetches\' cannot be a Tensor');
      }

      if (Array.isArray(arg1)) {
        if (arg1.length === 0) {
          throw new TypeError('\'fetches\' cannot be an empty array.');
        }
        isFetchesEmpty = false;
        // output names
        for (const name of arg1) {
          if (typeof name !== 'string') {
            throw new TypeError('\'fetches\' must be a string array or an object.');
          }
          if (this.outputNames.indexOf(name) === -1) {
            throw new RangeError(`'fetches' contains invalid output name: ${name}.`);
          }
          fetches[name] = null;
        }

        if (typeof arg2 === 'object' && arg2 !== null) {
          options = arg2;
        } else if (typeof arg2 !== 'undefined') {
          throw new TypeError('\'options\' must be an object.');
        }
      } else {
        // decide whether arg1 is fetches or options
        // if any output name is present and its value is valid OnnxValue, we consider it fetches
        let isFetches = false;
        const arg1Keys = Object.getOwnPropertyNames(arg1);
        for (const name of this.outputNames) {
          if (arg1Keys.indexOf(name) !== -1) {
            const v = (arg1 as InferenceSessionInterface.NullableOnnxValueMapType)[name];
            if (v === null || v instanceof Tensor) {
              isFetches = true;
              isFetchesEmpty = false;
              fetches[name] = v;
            }
          }
        }

        if (isFetches) {
          if (typeof arg2 === 'object' && arg2 !== null) {
            options = arg2;
          } else if (typeof arg2 !== 'undefined') {
            throw new TypeError('\'options\' must be an object.');
          }
        } else {
          options = arg1 as RunOptions;
        }
      }
    } else if (typeof arg1 !== 'undefined') {
      throw new TypeError('Unexpected argument[1]: must be \'fetches\' or \'options\'.');
    }

    // check if all inputs are in feed
    for (const name of this.inputNames) {
      if (typeof feeds[name] === 'undefined') {
        throw new Error(`input '${name}' is missing in 'feeds'.`);
      }
    }

    // if no fetches is specified, we use the full output names list
    if (isFetchesEmpty) {
      for (const name of this.outputNames) {
        fetches[name] = null;
      }
    }

    // feeds, fetches and options are prepared

    const results = await this.handler.run(feeds, fetches, options);
    const returnValue: {[name: string]: OnnxValue} = {};
    for (const key in results) {
      if (Object.hasOwnProperty.call(results, key)) {
        const result = results[key];
        if (result instanceof Tensor) {
          returnValue[key] = result;
        } else {
          returnValue[key] = new Tensor(result.type, result.data, result.dims);
        }
      }
    }
    return returnValue;
  }

  async release(): Promise<void> {
    return this.handler.dispose();
  }

  static create(path: string, options?: SessionOptions): Promise<InferenceSessionInterface>;
  static create(buffer: MaybePromise<ArrayBufferLike>, options?: SessionOptions): Promise<InferenceSessionInterface>;
  static create(
      buffer: MaybePromise<ArrayBufferLike>, byteOffset: number, byteLength?: number,
      options?: SessionOptions): Promise<InferenceSessionInterface>;
  static create(buffer: MaybePromise<Uint8Array>, options?: SessionOptions): Promise<InferenceSessionInterface>;
  static async create(
      arg0: string|MaybePromise<ArrayBufferLike|Uint8Array>, arg1?: SessionOptions|number, arg2?: number,
      arg3?: SessionOptions): Promise<InferenceSessionInterface> {
    let byteOffset: number|undefined;
    let byteLength: number|undefined;
    let options: SessionOptions|undefined;
    if (typeof arg1 === 'number') {
      byteOffset = arg1;
      byteLength = arg2;
      options = arg3;
    } else {
      options = arg1;
      byteOffset = byteLength = undefined;
    }
    if (typeof options === 'undefined') {
      options = {};
    }
    if (typeof options !== 'object' || options === null) {
      throw new TypeError('\'options\' must be an object.');
    }

    // either load from a file path (Node) / URL (web), or a buffer
    let stringOrUint8ArrayPromise: string|Promise<Uint8Array>;
    if (typeof arg0 === 'string') {
      stringOrUint8ArrayPromise = arg0;
      if (typeof arg1 === 'number') {
        throw new TypeError('\'options\' must be an object.');
      }
    } else {
      stringOrUint8ArrayPromise = Promise.resolve(arg0).then(buffer => {
        if (buffer instanceof Uint8Array) {
          if (typeof arg1 === 'number') {
            throw new TypeError('\'options\' must be an object.');
          }
          return buffer;
        }
        if (!(buffer instanceof ArrayBuffer ||
              (typeof SharedArrayBuffer !== 'undefined' && buffer instanceof SharedArrayBuffer))) {
          throw new TypeError('Unexpected argument[0]: must be \'path\' or \'buffer\'.');
        }
        if (typeof byteOffset === 'undefined') {
          byteOffset = 0;
        }
        if (typeof byteOffset !== 'number') {
          throw new TypeError('\'byteOffset\' must be a number.');
        }
        if (!Number.isSafeInteger(byteOffset)) {
          throw new RangeError('\'byteOffset\' must be an integer.');
        }
        if (byteOffset < 0 || byteOffset >= buffer.byteLength) {
          throw new RangeError(`'byteOffset' is out of range [0, ${buffer.byteLength}).`);
        }
        if (typeof byteLength === 'undefined') {
          byteLength = buffer.byteLength - byteOffset;
        }
        if (typeof byteLength !== 'number') {
          throw new TypeError('\'byteLength\' must be a number.');
        }
        if (!Number.isSafeInteger(byteLength)) {
          throw new RangeError('\'byteLength\' must be an integer.');
        }
        if (byteLength <= 0 || byteOffset + byteLength > buffer.byteLength) {
          throw new RangeError(`'byteLength' is out of range (0, ${buffer.byteLength - byteOffset}].`);
        }
        return new Uint8Array(buffer, byteOffset, byteLength);
      });
    }

    // get backend hints
    const eps = options.executionProviders || [];
    const backendHints = eps.map(i => typeof i === 'string' ? i : i.name);
    const backend = await resolveBackend(backendHints);
    const stringOrUint8Array = await Promise.resolve(stringOrUint8ArrayPromise);
    const handler = await backend.createInferenceSessionHandler(stringOrUint8Array, options);
    return new InferenceSession(handler);
  }

  startProfiling(): void {
    this.handler.startProfiling();
  }
  endProfiling(): void {
    this.handler.endProfiling();
  }

  get inputNames(): readonly string[] {
    return this.handler.inputNames;
  }
  get outputNames(): readonly string[] {
    return this.handler.outputNames;
  }

  private handler: InferenceSessionHandler;
}
