// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {env, InferenceSession, SessionHandler, TrainingSessionHandler} from 'onnxruntime-common';

import {SerializableModeldata} from './proxy-messages';
import {createSessionAllocate, initRuntime, isOrtEnvInitialized} from './wasm-core-impl';
import {createCheckpointHandle, createTrainingSessionHandle, releaseTrainingSessionAndCheckpoint} from './wasm-training-core-impl';

export class OnnxruntimeWebAssemblyTrainingSessionHandler implements TrainingSessionHandler {
  async loadParametersBuffer(_array: Uint8Array, _trainableOnly: boolean): Promise<void> {
    throw new Error('Method not implemented.');
  }
  async getContiguousParameters(_trainableOnly: boolean): Promise<Uint8Array> {
    throw new Error('Method not implemented.');
  }
  private sessionId: number;
  private checkpointId: number;

  inputNames: string[];
  outputNames: string[];

  inputEncodedNames: number[];
  outputEncodedNames: number[];

  async uriOrBufferToHeap(uriOrBuffer: string|Uint8Array): Promise<SerializableModeldata> {
    let buffer: Uint8Array;
    if (typeof uriOrBuffer === 'string') {
      const response = await fetch(uriOrBuffer);
      const arrayBuffer = await response.arrayBuffer();
      buffer = new Uint8Array(arrayBuffer);
    } else {
      buffer = uriOrBuffer;
    }
    return createSessionAllocate(buffer);
  }

  async createTrainingSession(
      checkpointStateUriOrBuffer: string|Uint8Array, trainModelUriOrBuffer: string|Uint8Array,
      evalModelUriOrBuffer: string|Uint8Array, optimizerModelUriOrBuffer: string|Uint8Array,
      options: InferenceSession.SessionOptions) {
    if (!isOrtEnvInitialized()) {
      await initRuntime(env);
    }
    const checkpointData: SerializableModeldata = await this.uriOrBufferToHeap(checkpointStateUriOrBuffer);
    const trainModelData: SerializableModeldata = await this.uriOrBufferToHeap(trainModelUriOrBuffer);
    // 0 is supposed to be the nullptr
    let evalModelData: SerializableModeldata = [0, 0];
    let optimizerModelData: SerializableModeldata = [0, 0];

    if (evalModelUriOrBuffer !== '') {
      evalModelData = await this.uriOrBufferToHeap(evalModelUriOrBuffer);
    }
    if (optimizerModelUriOrBuffer !== '') {
      optimizerModelData = await this.uriOrBufferToHeap(optimizerModelUriOrBuffer);
    }

    this.checkpointId = createCheckpointHandle(checkpointData);
    [[this.sessionId, this.inputNames, this.outputNames], this.inputEncodedNames, this.outputEncodedNames] =
        createTrainingSessionHandle(this.checkpointId, trainModelData, evalModelData, optimizerModelData, options);
  }

  async dispose(): Promise<void> {
    return releaseTrainingSessionAndCheckpoint(
        this.checkpointId, this.sessionId, this.inputEncodedNames, this.outputEncodedNames);
  }

  async runTrainStep(
      _feeds: SessionHandler.FeedsType, _fetches: SessionHandler.FetchesType,
      _options: InferenceSession.RunOptions): Promise<SessionHandler.ReturnType> {
    throw new Error('Method not implemented yet.');
  }
}
