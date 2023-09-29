// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TrainingSessionHandler} from './backend.js';
import {InferenceSession as InferenceSession} from './inference-session.js';
import {TrainingSession as TrainingSessionInterface, TrainingSessionCreateOptions} from './training-session.js';

type SessionOptions = InferenceSession.SessionOptions;

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

  static async create(_trainingOptions: TrainingSessionCreateOptions, _sessionOptions?: SessionOptions):
      Promise<TrainingSession> {
    throw new Error('Method not implemented');
  }

  async loadParametersBuffer(_array: Uint8Array, _trainableOnly: boolean): Promise<void> {
    throw new Error('Method not implemented.');
  }

  async getContiguousParameters(_trainableOnly: boolean): Promise<Uint8Array> {
    throw new Error('Method not implemented.');
  }

  runTrainStep(feeds: InferenceSession.OnnxValueMapType, options?: InferenceSession.RunOptions|undefined):
      Promise<InferenceSession.OnnxValueMapType>;
  runTrainStep(
      feeds: InferenceSession.OnnxValueMapType, fetches: InferenceSession.FetchesType,
      options?: InferenceSession.RunOptions|undefined): Promise<InferenceSession.OnnxValueMapType>;
  async runTrainStep(_feeds: unknown, _fetches?: unknown, _options?: unknown):
      Promise<InferenceSession.OnnxValueMapType> {
    throw new Error('Method not implemented.');
  }

  async release(): Promise<void> {
    return this.handler.dispose();
  }
}
