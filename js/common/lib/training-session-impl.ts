// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {resolveBackend} from './backend-impl.js';
import {TrainingSessionHandler} from './backend.js';
import {InferenceSession as InferenceSession} from './inference-session.js';
import {OnnxValue} from './onnx-value.js';
import {Tensor} from './tensor.js';
import {TrainingSession as TrainingSessionInterface, TrainingSessionCreateOptions} from './training-session.js';

type SessionOptions = InferenceSession.SessionOptions;
type FeedsType = InferenceSession.FeedsType;
type FetchesType = InferenceSession.FetchesType;
type ReturnType = InferenceSession.ReturnType;
type RunOptions = InferenceSession.RunOptions;

const noBackendErrMsg: string = 'Training backend could not be resolved. ' +
    'Make sure you\'re using the correct configuration & WebAssembly files.';

export class TrainingSession implements TrainingSessionInterface {
  private constructor(handler: TrainingSessionHandler) {
    this.handler = handler;
  }
  private handler: TrainingSessionHandler;

  get inputNames(): readonly string[] {
    return this.handler.inputNames;
  }
  get outputNames(): readonly string[] {
    return this.handler.outputNames;
  }

  static async create(trainingOptions: TrainingSessionCreateOptions, sessionOptions?: SessionOptions):
      Promise<TrainingSession> {
    const evalModel: string|Uint8Array = trainingOptions.evalModel || '';
    const optimizerModel: string|Uint8Array = trainingOptions.optimizerModel || '';
    const options: SessionOptions = sessionOptions || {};

    // get backend hints
    const eps = options.executionProviders || [];
    const backendHints = eps.map(i => typeof i === 'string' ? i : i.name);
    const backend = await resolveBackend(backendHints);
    if (backend.createTrainingSessionHandler) {
      const handler = await backend.createTrainingSessionHandler(
          trainingOptions.checkpointState, trainingOptions.trainModel, evalModel, optimizerModel, options);
      return new TrainingSession(handler);
    } else {
      throw new Error(noBackendErrMsg);
    }
  }

  runTrainStep(feeds: FeedsType, options?: RunOptions): Promise<ReturnType>;
  runTrainStep(feeds: FeedsType, fetches: FetchesType, options?: RunOptions): Promise<ReturnType>;
  async runTrainStep(feeds: FeedsType, arg1?: FetchesType|RunOptions, arg2?: RunOptions): Promise<ReturnType> {
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
            const v = (arg1 as InferenceSession.NullableOnnxValueMapType)[name];
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

    const results = await this.handler.runTrainStep(feeds, fetches, options);
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

  async getParametersSize(trainableOnly: boolean): Promise<number> {
    return this.handler.getParametersSize(trainableOnly);
  }

  async loadParametersBuffer(array: Float32Array, trainableOnly: boolean): Promise<void> {
    const paramsSize = await this.getParametersSize(trainableOnly);
    if (array.length !== paramsSize) {
      throw new Error('Size of the buffer passed into loadParametersBuffer must match the number of parameters in ' +
        'the model. Please use getParametersSize method to check.');
    }
    return this.handler.loadParametersBuffer(array, trainableOnly);
  }

  async getContiguousParameters(trainableOnly: boolean): Promise<OnnxValue> {
    return this.handler.getContiguousParameters(trainableOnly);
  }

  async release(): Promise<void> {
    return this.handler.dispose();
  }
}
