// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {resolveBackend} from './backend-impl.js';
import {TrainingSessionHandler} from './backend.js';
import {InferenceSession as InferenceSession} from './inference-session.js';
import {processModel, processModelOrOptions} from './session-impl-utils.js';
import {TrainingSession as TrainingSessionInterface} from './training-session.js';

type SessionOptions = InferenceSession.SessionOptions;

export class TrainingSession implements TrainingSessionInterface {
  private constructor(handler: TrainingSessionHandler) {
    this.handler = handler;
  }

  runTrainStep(feeds: InferenceSession.OnnxValueMapType, options?: InferenceSession.RunOptions):
      Promise<InferenceSession.OnnxValueMapType>;
  runTrainStep(
      feeds: InferenceSession.OnnxValueMapType, fetches: InferenceSession.FetchesType,
      options?: InferenceSession.RunOptions): Promise<InferenceSession.OnnxValueMapType>;
  runTrainStep(
      feeds: InferenceSession.OnnxValueMapType, arg1?: InferenceSession.FetchesType|InferenceSession.RunOptions,
      arg2?: InferenceSession.RunOptions): Promise<InferenceSession.OnnxValueMapType> {
    // using all arguments to prevent error from arising in transcompilation step about unused params
    if (arg1) console.log(arg1);
    if (arg2) console.log(arg2);
    throw new Error('Method not implemented.' + feeds);
  }
  release(): Promise<void> {
    return this.handler.dispose();
  }

  static create(checkpointStateUri: string, trainModelURI: string, options?: InferenceSession.SessionOptions):
      Promise<TrainingSession>;
  static create(
      checkpointState: string|ArrayBufferLike|Uint8Array, trainModelData: string|ArrayBufferLike|Uint8Array,
      evalModelData?: string|ArrayBufferLike|Uint8Array, optimizerModelData?: string|ArrayBufferLike|Uint8Array,
      options?: InferenceSession.SessionOptions): Promise<TrainingSession>;
  static async create(
      arg0: string|ArrayBufferLike|Uint8Array, arg1: string|ArrayBufferLike|Uint8Array,
      arg2?: string|ArrayBufferLike|Uint8Array|InferenceSession.SessionOptions,
      arg3?: string|ArrayBufferLike|Uint8Array, arg4?: InferenceSession.SessionOptions): Promise<TrainingSession> {
    // optional fields:
    let options: SessionOptions = {};
    let checkpointState: string|Uint8Array = processModel(arg0);
    let trainModel: string|Uint8Array = processModel(arg1);
    let evalModel: string|Uint8Array = '';
    let optimizerModel: string|Uint8Array = '';

    if (typeof arg2 !== 'undefined') {
      let [return1, return2] = processModelOrOptions(arg2, options, evalModel);
      options = return1;
      evalModel = return2;
    }
    if (typeof arg3 !== 'undefined') {
      optimizerModel = processModel(arg3);
    }
    if (typeof arg4 !== 'undefined') {
      options = arg4;
    }

    // get backend hints
    const eps = options.executionProviders || [];
    const backendHints = eps.map(i => typeof i === 'string' ? i : i.name);
    const backend = await resolveBackend(backendHints);
    const handler =
        await backend.createTrainingSessionHandler(checkpointState, trainModel, evalModel, optimizerModel, options);
    return new TrainingSession(handler);
  }

  inputNames: readonly string[];
  outputNames: readonly string[];

  private handler: TrainingSessionHandler;
}
