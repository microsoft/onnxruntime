// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { readFile } from 'fs';
import { env, InferenceSession, TrainingSessionHandler, Tensor } from 'onnxruntime-common';
import { promisify } from 'util';

import { SerializableModeldata } from './proxy-messages';
import { createSession, createSessionAllocate, createSessionFinalize, endProfiling, initializeRuntime, releaseSession, run } from './proxy-wrapper';

let runtimeInitialized: boolean;

export class OnnxruntimeWebAssemblyTrainingSessionHandler implements TrainingSessionHandler {
  private sessionId: number;
  private checkpointId: number;

  inputNames: string[];
  outputNames: string[];

  async dispose(): Promise<void> {
    return releaseSession(this.sessionId);
  }

  async runTrainStep(feeds: SessionHandler.FeedsType, fetches: SessionHandler.FetchesType, options: InferenceSession.RunOptions):
    Promise<SessionHandler.ReturnType> {

  }

  async runTrainStep(feeds: SessionHandler.FeedsType, options: InferenceSession.RunOptions):
    Promise<SessionHandler.ReturnType> {

  }
}
