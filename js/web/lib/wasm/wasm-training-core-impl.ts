// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession, Tensor} from 'onnxruntime-common';

import { prepareInputOutputTensor } from './wasm-core-impl';
import {SerializableModeldata, SerializableSessionMetadata, TensorMetadata} from './proxy-messages';
import {setSessionOptions} from './session-options';
import { tensorDataTypeEnumToString, tensorTypeToTypedArrayConstructor } from './wasm-common';
import {getInstance} from './wasm-factory';
import {checkLastError} from './wasm-utils';
import { setRunOptions } from './run-options';

const NO_TRAIN_FUNCS_MSG = 'Built without training APIs enabled. ' +
    'Make sure to use the onnxruntime-training package for training functionality.';

export const createCheckpointHandle = (checkpointData: SerializableModeldata): number => {
  const wasm = getInstance();

  let checkpointHandle = 0;

  try {
    if (wasm._OrtTrainingLoadCheckpoint) {
      checkpointHandle = wasm._OrtTrainingLoadCheckpoint(checkpointData[0], checkpointData[1]);
    } else {
      throw new Error(NO_TRAIN_FUNCS_MSG);
    }

    if (checkpointHandle === 0) {
      checkLastError('Error occurred when trying to create a CheckpointState.');
    }
    return checkpointHandle;
  } catch (e) {
    if (wasm._OrtTrainingReleaseCheckpoint && checkpointHandle !== 0) {
      wasm._OrtTrainingReleaseCheckpoint(checkpointHandle);
    }
    throw e;
  } finally {
    // free buffer from wasm heap
    wasm._OrtFree(checkpointData[0]);
  }
};

const getTrainingModelInputOutputCount = (trainingSessionId: number): [number, number] => {
  const wasm = getInstance();
  const stack = wasm.stackSave();
  try {
    const dataOffset = wasm.stackAlloc(8);
    if (wasm._OrtTrainingGetInputOutputCount) {
      const errorCode = wasm._OrtTrainingGetInputOutputCount(trainingSessionId, dataOffset, dataOffset + 4);
      if (errorCode !== 0) {
        checkLastError('Can\'t get session input/output count.');
      }
      return [wasm.HEAP32[dataOffset / 4], wasm.HEAP32[dataOffset / 4 + 1]];
    } else {
      throw new Error(NO_TRAIN_FUNCS_MSG);
    }
  } finally {
    wasm.stackRestore(stack);
  }
};

const getTrainingNamesLoop = (trainingSessionId: number, count: number, isInput: boolean): [string[], number[]] => {
  const names = [];
  const wasm = getInstance();

  const namesUTF8Encoded = [];

  for (let i = 0; i < count; i++) {
    if (wasm._OrtTrainingGetInputOutputName) {
      const name = wasm._OrtTrainingGetInputOutputName(trainingSessionId, i, isInput);
      if (name === 0) {
        checkLastError('Can\'t get input or output name');
      }

      namesUTF8Encoded.push(name);
      names.push(wasm.UTF8ToString(name));
    } else {
      throw new Error(NO_TRAIN_FUNCS_MSG);
    }
  }
  return [names, namesUTF8Encoded];
};

const getTrainingModelInputOutputNames = (trainingSessionId: number): [string[], number[], string[], number[]] => {
  const [inputCount, outputCount] = getTrainingModelInputOutputCount(trainingSessionId);

  const [inputNames, inputNamesUTF8Encoded] = getTrainingNamesLoop(trainingSessionId, inputCount, true);
  const [outputNames, outputNamesUTF8Encoded] = getTrainingNamesLoop(trainingSessionId, outputCount, false);

  return [inputNames, inputNamesUTF8Encoded, outputNames, outputNamesUTF8Encoded];
};

export const createTrainingSessionHandle =
    (checkpointHandle: number, trainModelData: SerializableModeldata, evalModelData: SerializableModeldata,
     optimizerModelData: SerializableModeldata,
     options: InferenceSession.SessionOptions): [SerializableSessionMetadata, number[], number[]] => {
      const wasm = getInstance();

      let trainingSessionHandle = 0;
      let sessionOptionsHandle = 0;
      let allocs: number[] = [];
      let inputNamesUTF8Encoded: number[] = [];
      let outputNamesUTF8Encoded: number[] = [];

      let inputNames: string[] = [];
      let outputNames: string[] = [];

      try {
        [sessionOptionsHandle, allocs] = setSessionOptions(options);
        if (wasm._OrtTrainingCreateSession) {
          trainingSessionHandle = wasm._OrtTrainingCreateSession(
              sessionOptionsHandle, checkpointHandle, trainModelData[0], trainModelData[1], evalModelData[0],
              evalModelData[1], optimizerModelData[0], optimizerModelData[1]);
        } else {
          throw new Error(NO_TRAIN_FUNCS_MSG);
        }

        if (trainingSessionHandle === 0) {
          checkLastError('Error occurred when trying to create a TrainingSession.');
        }

        [inputNames, inputNamesUTF8Encoded, outputNames, outputNamesUTF8Encoded] =
            getTrainingModelInputOutputNames(trainingSessionHandle);
        return [[trainingSessionHandle, inputNames, outputNames], inputNamesUTF8Encoded, outputNamesUTF8Encoded];

      } catch (e) {
        if (wasm._OrtTrainingReleaseSession && trainingSessionHandle !== 0) {
          wasm._OrtTrainingReleaseSession(trainingSessionHandle);
        }
        throw e;
      } finally {
        wasm._free(trainModelData[0]);
        wasm._free(evalModelData[0]);
        wasm._free(optimizerModelData[0]);

        if (sessionOptionsHandle !== 0) {
          wasm._OrtReleaseSessionOptions(sessionOptionsHandle);
        }
        allocs.forEach(alloc => wasm._free(alloc));
        inputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));
        outputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));
      }
    };

export const runTrainStep = async(
  trainingSessionId: number, inputIndices: number[], inputTensors: TensorMetadata[], outputIndices: number[],
  outputTensors: Array<TensorMetadata|null>, options: InferenceSession.RunOptions): Promise<TensorMetadata[]> => {
    const wasm = getInstance();

  const inputCount = inputIndices.length;
  const outputCount = outputIndices.length;

  let runOptionsHandle = 0;
  let runOptionsAllocs: number[] = [];

  const inputTensorHandles: number[] = [];
  const outputTensorHandles: number[] = [];
  const inputOutputAllocs: number[] = [];

  const beforeRunStack = wasm.stackSave();
  const inputValuesOffset = wasm.stackAlloc(inputCount * 4);
  const outputValuesOffset = wasm.stackAlloc(outputCount * 4);

  try {
    [runOptionsHandle, runOptionsAllocs] = setRunOptions(options);

    // TODO:
    // move all input and output processing -> wasm heap to one helper method????
    // can abstract out the similarities between input and output
    // create input tensors
    for (let i = 0; i < inputCount; i++) {
      prepareInputOutputTensor(inputTensors[i], inputTensorHandles, inputOutputAllocs, trainingSessionId, inputIndices[i]);
    }

    // create output tensors
    for (let i = 0; i < outputCount; i++) {
      prepareInputOutputTensor(
          outputTensors[i], outputTensorHandles, inputOutputAllocs, trainingSessionId, inputCount + outputIndices[i]);
    }

    let inputValuesIndex = inputValuesOffset / 4;
    let outputValuesIndex = outputValuesOffset / 4;
    for (let i = 0; i < inputCount; i++) {
      wasm.HEAPU32[inputValuesIndex++] = inputTensorHandles[i];
    }
    for (let i = 0; i < outputCount; i++) {
      wasm.HEAPU32[outputValuesIndex++] = outputTensorHandles[i];
    }

    let errorCode: number;

    if (wasm._OrtTrainingRunTrainStep) {
      errorCode = await wasm._OrtTrainingRunTrainStep(trainingSessionId, inputValuesOffset, inputCount,
        outputValuesOffset, outputCount, runOptionsHandle);
    }
    else {
      throw new Error(NO_TRAIN_FUNCS_MSG);
    }

    if (errorCode !== 0) {
      checkLastError('failed to call OrtTrainingRunTrainStep in the WebAssembly layer');
    }

    const output: TensorMetadata[] = [];

    for (let i = 0; i < outputCount; i++) {
      const tensor = wasm.HEAPU32[outputValuesOffset / 4 + i];

      const beforeGetTensorDataStack = wasm.stackSave();
      // stack allocate 4 pointer value
      const tensorDataOffset = wasm.stackAlloc(4 * 4);

      let keepOutputTensor = false;
      let type: Tensor.Type|undefined, dataOffset = 0;
      try {
        const errorCode = wasm._OrtGetTensorData(
            tensor, tensorDataOffset, tensorDataOffset + 4, tensorDataOffset + 8, tensorDataOffset + 12);
        if (errorCode !== 0) {
          checkLastError(`Can't access output tensor data on index ${i}.`);
        }
        let tensorDataIndex = tensorDataOffset / 4;
        const dataType = wasm.HEAPU32[tensorDataIndex++];
        dataOffset = wasm.HEAPU32[tensorDataIndex++];
        const dimsOffset = wasm.HEAPU32[tensorDataIndex++];
        const dimsLength = wasm.HEAPU32[tensorDataIndex++];
        const dims = [];
        for (let i = 0; i < dimsLength; i++) {
          dims.push(wasm.HEAPU32[dimsOffset / 4 + i]);
        }
        wasm._OrtFree(dimsOffset);

        const size = dims.reduce((a, b) => a * b, 1);
        type = tensorDataTypeEnumToString(dataType);

        if (type === 'string') {
          const stringData: string[] = [];
          let dataIndex = dataOffset / 4;
          for (let i = 0; i < size; i++) {
            const offset = wasm.HEAPU32[dataIndex++];
            const maxBytesToRead = i === size - 1 ? undefined : wasm.HEAPU32[dataIndex] - offset;
            stringData.push(wasm.UTF8ToString(offset, maxBytesToRead));
          }
          output.push([type, dims, stringData, 'cpu']);
        } else {
            const typedArrayConstructor = tensorTypeToTypedArrayConstructor(type);
            const data = new typedArrayConstructor(size);
            new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
                .set(wasm.HEAPU8.subarray(dataOffset, dataOffset + data.byteLength));
            output.push([type, dims, data, 'cpu']);
        }
      } finally {
        wasm.stackRestore(beforeGetTensorDataStack);
        if (type === 'string' && dataOffset) {
          wasm._free(dataOffset);
        }
        if (!keepOutputTensor) {
          wasm._OrtReleaseTensor(tensor);
        }
      }
    }

    return output;
  } finally {
    wasm.stackRestore(beforeRunStack);

    inputTensorHandles.forEach(v => wasm._OrtReleaseTensor(v));
    outputTensorHandles.forEach(v => wasm._OrtReleaseTensor(v));
    inputOutputAllocs.forEach(p => wasm._free(p));

    if (runOptionsHandle !== 0) {
      wasm._OrtReleaseRunOptions(runOptionsHandle);
    }
    runOptionsAllocs.forEach(p => wasm._free(p));
  }
};

export const releaseTrainingSessionAndCheckpoint =
    (checkpointId: number, sessionId: number, inputNamesUTF8Encoded: number[], outputNamesUTF8Encoded: number[]):
        void => {
          const wasm = getInstance();
          inputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));
          outputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));

          if (wasm._OrtTrainingReleaseCheckpoint) {
            wasm._OrtTrainingReleaseCheckpoint(checkpointId);
          }
          if (wasm._OrtTrainingReleaseSession) {
            wasm._OrtTrainingReleaseSession(sessionId);
          }
        };
