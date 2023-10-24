// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {env, InferenceSession, OnnxValue, SessionHandler, Tensor, TrainingSessionHandler} from 'onnxruntime-common';

import {SerializableModeldata} from './proxy-messages';
import {decodeTensorMetadata, encodeTensorMetadata} from './session-handler-inference';
import {createSessionAllocate, initRuntime, isOrtEnvInitialized} from './wasm-core-impl';
import {createCheckpointHandle, createTrainingSessionHandle, getContiguousParameters, getParametersSize, loadParametersBuffer, releaseTrainingSessionAndCheckpoint, runTrainStep} from './wasm-training-core-impl';

export class OnnxruntimeWebAssemblyTrainingSessionHandler implements TrainingSessionHandler {
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

  async runTrainStep(
      feeds: SessionHandler.FeedsType, fetches: SessionHandler.FetchesType,
      options: InferenceSession.RunOptions): Promise<SessionHandler.ReturnType> {
    const inputArray: Tensor[] = [];
    const inputIndices: number[] = [];
    Object.entries(feeds).forEach(kvp => {
      const name = kvp[0];
      const tensor = kvp[1];
      const index = this.inputNames.indexOf(name);
      if (index === -1) {
        throw new Error(`invalid input '${name}'`);
      }
      inputArray.push(tensor);
      inputIndices.push(index);
    });

    const outputArray: Array<Tensor|null> = [];
    const outputIndices: number[] = [];
    Object.entries(fetches).forEach(kvp => {
      const name = kvp[0];
      const tensor = kvp[1];
      const index = this.outputNames.indexOf(name);
      if (index === -1) {
        throw new Error(`invalid output '${name}'`);
      }
      outputArray.push(tensor);
      outputIndices.push(index);
    });

    const inputs =
        inputArray.map((t, i) => encodeTensorMetadata(t, () => `input "${this.inputNames[inputIndices[i]]}"`));
    const outputs = outputArray.map(
        (t, i) => t ? encodeTensorMetadata(t, () => `output "${this.outputNames[outputIndices[i]]}"`) : null);

    const results = await runTrainStep(this.sessionId, inputIndices, inputs, outputIndices, outputs, options);

    const resultMap: SessionHandler.ReturnType = {};
    for (let i = 0; i < results.length; i++) {
      resultMap[this.outputNames[outputIndices[i]]] = outputArray[i] ?? decodeTensorMetadata(results[i]);
    }
    return resultMap;
  }

  async getParametersSize(trainableOnly: boolean): Promise<number> {
    return getParametersSize(this.sessionId, trainableOnly);
  }

  async loadParametersBuffer(array: Float32Array, trainableOnly: boolean): Promise<void> {
    await loadParametersBuffer(this.sessionId, array, trainableOnly);
  }
  async getContiguousParameters(trainableOnly: boolean): Promise<OnnxValue> {
    const tensorResult = await getContiguousParameters(this.sessionId, trainableOnly);
    return decodeTensorMetadata(tensorResult);
  }

  async dispose(): Promise<void> {
    return releaseTrainingSessionAndCheckpoint(
        this.checkpointId, this.sessionId, this.inputEncodedNames, this.outputEncodedNames);
  }
}
