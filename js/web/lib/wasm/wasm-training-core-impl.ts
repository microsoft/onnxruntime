// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession} from 'onnxruntime-common';

import {SerializableModeldata, SerializableSessionMetadata} from './proxy-messages';
import {setSessionOptions} from './session-options';
import {getInstance} from './wasm-factory';
import {checkLastError} from './wasm-utils';

const NO_TRAIN_FUNCS_MSG =
    'Built without training API\'s enabled. Use the onnxruntime-web/training import for training ' +
    'functionality, and make sure that all the correct artifacts are built & moved to the correct folder if ' +
    'using a custom build. Check https://onnxruntime.ai/docs/build/web.html for more information.';

export const createCheckpointHandle = (checkpointData: SerializableModeldata): number => {
  const wasm = getInstance();

  const [checkpointDataOffset, checkpointDataLength] = checkpointData;
  let checkpointHandle = 0;

  try {
    if (wasm._OrtTrainingLoadCheckpoint) {
      checkpointHandle = wasm._OrtTrainingLoadCheckpoint(checkpointDataOffset, checkpointDataLength);
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

const getModelInputOutputCount = (trainingSessionId: number, IsEvalModel: boolean): [number, number] => {
  const wasm = getInstance();
  const stack = wasm.stackSave();
  try {
    const dataOffset = wasm.stackAlloc(8);
    if (wasm._OrtTrainingGetInputOutputCount) {
      const errorCode = wasm._OrtTrainingGetInputOutputCount(trainingSessionId, dataOffset, dataOffset + 4, false);
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

const getModelInputOutputNamesLoop = (trainingSessionId: number, count: number, isInput: boolean, IsEvalModel:boolean): [string[], number[]] => {
  const names = [];
  const wasm = getInstance();

  const namesUTF8Encoded = [];

  for (let i = 0; i < count; i++) {
    if (wasm._OrtTrainingGetInputOutputName) {
      const name = wasm._OrtTrainingGetInputOutputName(trainingSessionId, i, isInput, false);
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
