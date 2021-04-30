function _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }

function _classPrivateFieldGet(receiver, privateMap) { var descriptor = privateMap.get(receiver); if (!descriptor) { throw new TypeError("attempted to get private field on non-instance"); } if (descriptor.get) { return descriptor.get.call(receiver); } return descriptor.value; }

function _classPrivateFieldSet(receiver, privateMap, value) { var descriptor = privateMap.get(receiver); if (!descriptor) { throw new TypeError("attempted to set private field on non-instance"); } if (descriptor.set) { descriptor.set.call(receiver, value); } else { if (!descriptor.writable) { throw new TypeError("attempted to set read only private field"); } descriptor.value = value; } return value; }

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import { Tensor } from 'onnxruntime-common';
import { binding } from './binding';
import { Buffer } from 'buffer';

const tensorTypeToTypedArray = type => {
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
      throw new Error("unsupported type: ".concat(type));
  }
};

var _inferenceSession = new WeakMap();

var _key = new WeakMap();

class OnnxruntimeSessionHandler {
  constructor(path) {
    _inferenceSession.set(this, {
      writable: true,
      value: void 0
    });

    _key.set(this, {
      writable: true,
      value: void 0
    });

    _defineProperty(this, "inputNames", void 0);

    _defineProperty(this, "outputNames", void 0);

    _classPrivateFieldSet(this, _inferenceSession, binding);

    _classPrivateFieldSet(this, _key, path);

    this.inputNames = [];
    this.outputNames = [];
  }

  async loadModel(options) {
    try {
      // load a model
      const results = await _classPrivateFieldGet(this, _inferenceSession).loadModel(_classPrivateFieldGet(this, _key), options); // resolve promise if onnxruntime session is successfully created

      if (results.key !== _classPrivateFieldGet(this, _key)) {
        throw new Error('Session key is invalid');
      }

      this.inputNames = results.inputNames;
      this.outputNames = results.outputNames;
      return Promise.resolve();
    } catch (e) {
      throw new Error("Can't load a model");
    }
  }

  async dispose() {
    return Promise.resolve();
  }

  startProfiling() {// TODO: implement profiling
  }

  endProfiling() {// TODO: implement profiling
  }

  async run(feeds, fetches, options) {
    return new Promise(async (resolve, reject) => {
      try {
        // Java API doesn't support preallocated output names and allows only string array as parameter.
        const outputNames = [];

        for (const fetch in fetches) {
          outputNames.push(fetch);
        }

        const input = this.encodeFeedsType(feeds);
        const results = await _classPrivateFieldGet(this, _inferenceSession).run(_classPrivateFieldGet(this, _key), input, outputNames, options);
        const output = this.decodeReturnType(results);
        resolve(output);
      } catch (e) {
        reject(e);
      }
    });
  }

  encodeFeedsType(feeds) {
    const returnValue = {};

    for (const key in feeds) {
      if (Object.hasOwnProperty.call(feeds, key)) {
        let data;

        if (Array.isArray(feeds[key].data)) {
          data = feeds[key].data;
        } else {
          // Base64-encode tensor data
          data = Buffer.from(feeds[key].data.buffer).toString('base64');
        }

        returnValue[key] = {
          dims: feeds[key].dims,
          type: feeds[key].type,
          data
        };
      }
    }

    return returnValue;
  }

  decodeReturnType(results) {
    const returnValue = {};

    for (const key in results) {
      if (Object.hasOwnProperty.call(results, key)) {
        let tensorData;

        if (Array.isArray(results[key].data)) {
          tensorData = results[key].data;
        } else {
          const buffer = Buffer.from(results[key].data, 'base64');
          const typedArray = tensorTypeToTypedArray(results[key].type);
          tensorData = new typedArray(buffer.buffer, buffer.byteOffset, buffer.length / typedArray.BYTES_PER_ELEMENT);
        }

        returnValue[key] = new Tensor(results[key].type, tensorData, results[key].dims);
      }
    }

    return returnValue;
  }

}

class OnnxruntimeBackend {
  async init() {
    return Promise.resolve();
  }

  async createSessionHandler(pathOrBuffer, options) {
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
//# sourceMappingURL=backend.js.map