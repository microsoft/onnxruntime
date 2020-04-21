import {binding, Binding} from './binding';
import {InferenceSession as InferenceSessionInterface, InferenceSessionFactory} from './inference-session';
import {OnnxValue} from './onnx-value';
import {Tensor} from './tensor';

type SessionOptions = InferenceSessionInterface.SessionOptions;
type RunOptions = InferenceSessionInterface.RunOptions;
type FeedsType = InferenceSessionInterface.FeedsType;
type FetchesType = InferenceSessionInterface.FetchesType;
type ReturnType = InferenceSessionInterface.ReturnType;

class InferenceSession implements InferenceSessionInterface {
  constructor(session: Binding.InferenceSession) {
    this.#session = session;
    // cache metadata
    this.inputNames = this.#session.inputNames;
    this.outputNames = this.#session.outputNames;
  }
  run(feeds: FeedsType, options?: RunOptions): Promise<ReturnType>;
  run(feeds: FeedsType, fetches: FetchesType, options?: RunOptions): Promise<ReturnType>;
  run(feeds: FeedsType, arg1?: FetchesType|RunOptions, arg2?: RunOptions): Promise<ReturnType> {
    let fetches: {[name: string]: OnnxValue|null} = {};
    let options: RunOptions = {};
    // check inputs
    if (typeof (feeds) !== 'object' || feeds instanceof Tensor || Array.isArray(feeds)) {
      throw new TypeError(
          'feeds must be an object that use input names as keys and OnnxValue as corresponding values.');
    }

    let isFetchesEmpty = true;
    // determine which override is being used
    if (typeof (arg1) === 'object') {
      if (Array.isArray(arg1)) {
        isFetchesEmpty = false;
        // output names
        for (const name of arg1) {
          if (typeof (name) !== 'string') {
            throw new TypeError('fetches must be a string array or an object.');
          }
          fetches[name] = null;
        }
      } else {
        // decide whether arg1 is fetches or options
        // if any output name is present and its value is valid OnnxValue, we consider it fetches
        let isFetches = false;
        const arg1Keys = Object.getOwnPropertyNames(arg1);
        for (const name of this.outputNames) {
          if (arg1Keys.indexOf(name) !== -1) {
            const v = arg1[name];
            if (v === null || v.constructor.name === 'Tensor') {
              isFetches = true;
              isFetchesEmpty = false;
              fetches[name] = v;
            }
          }
        }

        if (isFetches) {
          if (typeof (arg2) === 'object') {
            options = arg2;
          } else if (typeof (arg2) !== 'undefined') {
            throw new TypeError('options must be an object.');
          }
        } else {
          options = arg1 as RunOptions;
        }
      }
    } else if (typeof (arg1) !== 'undefined') {
      throw new TypeError('unexpected argument. arg1 must be fetches or options');
    }

    // check if all inputs are in feed
    for (const name of this.inputNames) {
      if (feeds[name] === undefined) {
        throw new Error(`input '${name}' is missing in feeds.`)
      }
    }

    // if no fetches is specified, we use the full output names list
    if (isFetchesEmpty) {
      for (const name of this.outputNames) {
        fetches[name] = null;
      }
    }

    // feeds, fetches and options are prepared

    // promise start here
    //
    //
    return new Promise((resolve, reject) => {
      process.nextTick(() => {
        try {
          const returnValue: {[name: string]: OnnxValue} = {};
          const results = this.#session.run(feeds, fetches, options);
          for (const key in results) {
            returnValue[key] = new Tensor(results[key].type, results[key].data, results[key].dims);
          }
          resolve(returnValue);
        } catch (e) {
          // reject if any error is thrown
          reject(e);
        }
      });
    });
  }

  inputNames: readonly string[];
  outputNames: readonly string[];

  #session: Binding.InferenceSession;
}

export const impl: InferenceSessionFactory = {
  create: function(
      arg0: string|ArrayBufferLike|Uint8Array, arg1?: SessionOptions|number, arg2?: number,
      arg3?: SessionOptions): Promise<InferenceSession> {
    // either load from a file or buffer
    let loadFromFilePath = false;
    let filePath: string;
    let buffer: ArrayBufferLike;
    let byteOffset: number = -1;
    let byteLength: number = -1;
    let options: SessionOptions = {};

    if (typeof (arg0) === 'string') {
      loadFromFilePath = true;
      filePath = arg0;
      if (typeof (arg1) === 'object') {
        options = arg1;
      } else if (typeof (arg1) !== 'undefined') {
        throw new TypeError('options must be an object.');
      }
    } else if (arg0 instanceof Uint8Array) {
      buffer = arg0.buffer;
      byteOffset = arg0.byteOffset;
      byteLength = arg0.byteLength;
      if (typeof (arg1) === 'object') {
        options = arg1;
      } else if (typeof (arg1) !== 'undefined') {
        throw new TypeError('options must be an object.');
      }
    } else if (arg0 instanceof ArrayBuffer || arg0 instanceof SharedArrayBuffer) {
      buffer = arg0;
      if (typeof (arg1) === 'object') {
        options = arg1;
      } else if (typeof (arg1) === 'number') {
        byteOffset = arg1;
        if (typeof (arg2) === 'number') {
          byteLength = arg2;
          if (typeof (arg3) === 'object') {
            options = arg3;
          } else if (typeof (arg3) !== 'undefined') {
            throw new TypeError('options must be an object.');
          }
        } else if (typeof (arg2) !== 'undefined') {
          throw new TypeError('byteLength must be a number.');
        }
      } else if (typeof (arg1) !== 'undefined') {
        throw new TypeError('options must be an object.');
      }
    }

    if (!loadFromFilePath &&
        (!Number.isSafeInteger(byteOffset) || !Number.isSafeInteger(byteLength) || byteOffset < 0 || byteLength < 0)) {
      throw new RangeError('byteOffset and byteLength must be positive integers.');
    }

    // promise start here
    //
    //
    return new Promise((resolve, reject) => {
      process.nextTick(() => {
        try {
          // create native session wrapper
          const sessionWrapper = new binding.InferenceSession(options);
          // load model
          if (loadFromFilePath) {
            sessionWrapper.loadModel(filePath);
          } else {
            sessionWrapper.loadModel(buffer, byteOffset, byteLength);
          }
          // resolve promise if created successfully
          resolve(new InferenceSession(sessionWrapper));
        } catch (e) {
          // reject if any error is thrown
          reject(e);
        }
      });
    });
  }
};
