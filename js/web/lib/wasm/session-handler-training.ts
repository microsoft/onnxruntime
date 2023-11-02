// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {env, InferenceSession, OnnxValue, SessionHandler, Tensor, TrainingSessionHandler} from 'onnxruntime-common';

import {SerializableModeldata, TensorMetadata} from './proxy-messages';
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

  /**
   * Helper method that converts a feeds or fetches datatype to two arrays, one of values and one that stores the
   * corresponding name as a number referring to the index in the list of names provided.
   *
   * @param feeds meant to match either SessionHandler.FeedsType or SessionHandler.FetchesType
   * @param names either inputNames or outputNames
   * @returns a tuple of a list of values and a list of indices.
   */
  convertMapIntoValuesArrayAndIndicesArray<T, U>(
      feeds: {[name: string]: T}, names: string[], mapFunc: (val: T, index: number) => U): [T[], number[], U[]] {
    const values: T[] = [];
    const indices: number[] = [];
    Object.entries(feeds).forEach(kvp => {
      const name = kvp[0];
      const tensor = kvp[1];
      const index = names.indexOf(name);
      if (index === -1) {
        throw new Error(`invalid input '${name}`);
      }
      values.push(tensor);
      indices.push(index);
    });

    const uList = values.map(mapFunc);
    return [values, indices, uList];
  }

  /**
   * Helper method that converts the TensorMetadata that the wasm-core functions return to the
   * SessionHandler.ReturnType. Any outputs in the provided outputArray that are falsy will be populated with the
   * corresponding result.
   *
   * @param results used to populate the resultMap if there is no value for that outputName already
   * @param outputArray used to populate the resultMap. If null or undefined, use the corresponding result from results
   * @param outputIndices specifies which outputName the corresponding value for outputArray refers to.
   * @returns a map of output names and OnnxValues.
   */
  convertTensorMetadataToReturnType(
      results: TensorMetadata[], outputArray: Array<Tensor|null>, outputIndices: number[]): SessionHandler.ReturnType {
    const resultMap: SessionHandler.ReturnType = {};
    for (let i = 0; i < results.length; i++) {
      resultMap[this.outputNames[outputIndices[i]]] = outputArray[i] ?? decodeTensorMetadata(results[i]);
    }
    return resultMap;
  }

  async runTrainStep(
      feeds: SessionHandler.FeedsType, fetches: SessionHandler.FetchesType,
      options: InferenceSession.RunOptions): Promise<SessionHandler.ReturnType> {
    const [, inputIndices, inputs] = this.convertMapIntoValuesArrayAndIndicesArray<Tensor, TensorMetadata>(
        feeds, this.inputNames,
        (t, i): TensorMetadata => encodeTensorMetadata(t, () => `input "${this.inputNames[inputIndices[i]]}"`));

    const [outputArray, outputIndices, outputs] =
        this.convertMapIntoValuesArrayAndIndicesArray<Tensor|null, TensorMetadata|null>(
            fetches, this.outputNames,
            (t, i): TensorMetadata|null =>
                t ? encodeTensorMetadata(t, () => `output "${this.outputNames[outputIndices[i]]}"`) : null);

    const results = await runTrainStep(this.sessionId, inputIndices, inputs, outputIndices, outputs, options);
    return this.convertTensorMetadataToReturnType(results, outputArray, outputIndices);
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
