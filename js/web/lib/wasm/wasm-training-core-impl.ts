// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession, Tensor} from 'onnxruntime-common';

import {SerializableModeldata, SerializableSessionMetadata, TensorMetadata} from './proxy-messages';
import {setRunOptions} from './run-options';
import {setSessionOptions} from './session-options';
import {tensorDataTypeEnumToString, tensorTypeToTypedArrayConstructor} from './wasm-common';
import {prepareInputOutputTensor} from './wasm-core-impl';
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

const getModelInputOutputCount = (trainingSessionId: number, isEvalModel: boolean): [number, number] => {
  const wasm = getInstance();
  const stack = wasm.stackSave();
  try {
    const dataOffset = wasm.stackAlloc(8);
    if (wasm._OrtTrainingGetModelInputOutputCount) {
      const errorCode =
          wasm._OrtTrainingGetModelInputOutputCount(trainingSessionId, dataOffset, dataOffset + 4, isEvalModel);
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

const getModelInputOutputNamesLoop =
    (trainingSessionId: number, count: number, isInput: boolean, isEvalModel: boolean): [string[], number[]] => {
      const names = [];
      const wasm = getInstance();

      const namesUTF8Encoded = [];

      for (let i = 0; i < count; i++) {
        if (wasm._OrtTrainingGetModelInputOutputName) {
          const name = wasm._OrtTrainingGetModelInputOutputName(trainingSessionId, i, isInput, isEvalModel);
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
  const [inputCount, outputCount] = getModelInputOutputCount(trainingSessionId, false);

  const [inputNames, inputNamesUTF8Encoded] = getModelInputOutputNamesLoop(trainingSessionId, inputCount, true, false);
  const [outputNames, outputNamesUTF8Encoded] =
      getModelInputOutputNamesLoop(trainingSessionId, outputCount, false, false);

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

/**
 * Prepares input and output tensors by creating the tensors in the WASM side then creates a list of the handles of the
 * WASM tensors.
 *
 * @param trainingSessionId
 * @param indices for each tensor, the index of the input or output name that the tensor corresponds with
 * @param tensors list of TensorMetaData
 * @param tensorHandles should pass in an empty list of numbers; modified in-place by this method & stores the resulting
 *                      handles of the allocated tensors on the heap
 * @param inputOutputAllocs modified in-place by this method
 * @param indexAdd constant to add to the index that is passed to prepareInputOutputTensor
 */
const createAndAllocateTensors =
    (trainingSessionId: number, indices: number[], tensors: Array<TensorMetadata|null>, tensorHandles: number[],
     inputOutputAllocs: number[], indexAdd: number) => {
      const wasm = getInstance();

      const count = indices.length;
      const valuesOffset = wasm.stackAlloc(count * 4);

      // creates the tensors
      for (let i = 0; i < count; i++) {
        prepareInputOutputTensor(
            tensors[i], tensorHandles, inputOutputAllocs, trainingSessionId, indexAdd + indices[i]);
      }

      // moves to heap
      let valuesIndex = valuesOffset / 4;
      for (let i = 0; i < count; i++) {
        wasm.HEAPU32[valuesIndex++] = tensorHandles[i];
      }

      return valuesOffset;
    };

/**
 * Retrieves the information from the output tensor handles, copies to an array, and frees the WASM information
 * associated with the tensor handle.
 *
 * @param outputValuesOffset
 * @param outputCount
 * @returns list of TensorMetadata retrieved from the output handles.
 */
const moveOutputToTensorMetadataArr =
    (outputValuesOffset: number, outputCount: number, outputTensorHandles: number[],
     outputTensors: Array<TensorMetadata|null>) => {
      const wasm = getInstance();
      const output: TensorMetadata[] = [];

      for (let i = 0; i < outputCount; i++) {
        const tensor = wasm.HEAPU32[outputValuesOffset / 4 + i];
        if (tensor === outputTensorHandles[i]) {
          // output tensor is pre-allocated. no need to copy data.
          output.push(outputTensors[i]!);
          continue;
        }

        const beforeGetTensorDataStack = wasm.stackSave();
        // stack allocate 4 pointer value
        const tensorDataOffset = wasm.stackAlloc(4 * 4);

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
        }
      }

      return output;
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

  try {
    // prepare parameters by moving them to heap
    [runOptionsHandle, runOptionsAllocs] = setRunOptions(options);

    // handle inputs -- you don't want anything added to the index
    const inputValuesOffset = createAndAllocateTensors(
        trainingSessionId, inputIndices, inputTensors, inputTensorHandles, inputOutputAllocs, 0);
    // handle outputs
    // you want inputCount to be added to the index of every output tensor passed to prepareInputOutputTensor
    const outputValuesOffset = createAndAllocateTensors(
        trainingSessionId, outputIndices, outputTensors, outputTensorHandles, inputOutputAllocs, inputCount);

    if (wasm._OrtTrainingRunTrainStep) {
      const errorCode = wasm._OrtTrainingRunTrainStep(
          trainingSessionId, inputValuesOffset, inputCount, outputValuesOffset, outputCount, runOptionsHandle);

      if (errorCode !== 0) {
        checkLastError('failed to call OrtTrainingRunTrainStep in the WebAssembly layer');
      }
    } else {
      throw new Error(NO_TRAIN_FUNCS_MSG);
    }

    return moveOutputToTensorMetadataArr(outputValuesOffset, outputCount, outputTensorHandles, outputTensors);
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

          if (wasm._OrtTrainingReleaseSession) {
            wasm._OrtTrainingReleaseSession(sessionId);
          }
          if (wasm._OrtTrainingReleaseCheckpoint) {
            wasm._OrtTrainingReleaseCheckpoint(checkpointId);
          }
        };
