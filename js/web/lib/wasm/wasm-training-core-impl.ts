// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession, Tensor} from 'onnxruntime-common';

import {SerializableModeldata, SerializableSessionMetadata, TensorMetadata} from './proxy-messages';
import {setRunOptions} from './run-options';
import {setSessionOptions} from './session-options';
import {dataLocationStringToEnum, tensorDataTypeEnumToString, tensorDataTypeStringToEnum, tensorTypeToTypedArrayConstructor} from './wasm-common';
import {prepareInputOutputTensor} from './wasm-core-impl';
import {getInstance} from './wasm-factory';
import {checkLastError} from './wasm-utils';

const NO_TRAIN_FUNCS_MSG =
    'Built without training API\'s enabled. Use the onnxruntime-web/training import for training ' +
    'functionality, and make sure that all the correct artifacts are built & moved to the correct folder if ' +
    'using a custom build. Check https://onnxruntime.ai/docs/build/web.html for more information.';

/**
 * Runs the checkLastError function which will throw an error, if the provided error code matches the specified
 * pattern for an error code.
 * @param errCode number to evaluated for if it's an error
 * @param message message to pass into checkLastError
 * @param checkNeqZero when true, treats not equal to zero as an error.
 *                     When false, treats equal to zero as an error.
 */
const ifErrCodeCheckLastError = (errCode: number, message: string, checkNeqZero = true) => {
  if (checkNeqZero && errCode !== 0) {
    checkLastError(message);
  } else if (!checkNeqZero && errCode === 0) {
    checkLastError(message);
  }
};

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

    ifErrCodeCheckLastError(checkpointHandle, 'Error occurred when trying to create a CheckpointState', false);
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
      ifErrCodeCheckLastError(errorCode, 'Can\'t get session input/output count.');
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
          ifErrCodeCheckLastError(name, `Can't get input or output name -- is input: ${isInput}, index ${i}`, false);

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

        ifErrCodeCheckLastError(trainingSessionHandle, 'Error occurred when trying to create a TrainingSession', false);

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
      const count = indices.length;

      // creates the tensors
      for (let i = 0; i < count; i++) {
        prepareInputOutputTensor(
            tensors[i], tensorHandles, inputOutputAllocs, trainingSessionId, indexAdd + indices[i]);
      }

      // moves to heap
      const wasm = getInstance();
      const valuesOffset = wasm.stackAlloc(count * 4);
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
          ifErrCodeCheckLastError(errorCode, `Can't access output tensor data on index ${i}.`);

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
          wasm._OrtReleaseTensor(tensor);
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
      ifErrCodeCheckLastError(errorCode, 'failed to call OrtTrainingRunTrainStep in the WebAssembly layer');
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

export const getParametersSize = (trainingSessionId: number, trainableOnly: boolean): number => {
  const wasm = getInstance();
  const stack = wasm.stackSave();

  try {
    const sizeOffset = wasm.stackAlloc(4);
    if (wasm._OrtTrainingGetParametersSize) {
      const errorCode = wasm._OrtTrainingGetParametersSize(trainingSessionId, sizeOffset, trainableOnly);
      ifErrCodeCheckLastError(errorCode, 'Can\'t get parameters size');

      return wasm.HEAP32[sizeOffset / 4];
    } else {
      throw new Error(NO_TRAIN_FUNCS_MSG);
    }
  } finally {
    wasm.stackRestore(stack);
  }
};

export const getContiguousParameters =
    async(trainingSessionId: number, trainableOnly: boolean): Promise<TensorMetadata> => {
  const wasm = getInstance();
  const stack = wasm.stackSave();

  const tensorTypeAsString = 'float32';
  const locationAsString = 'cpu';

  const parametersSize = getParametersSize(trainingSessionId, trainableOnly);
  let tensor = 0;

  // allocates a buffer of the correct size on the WASM heap
  const paramsByteLength = 4 * parametersSize;
  const paramsOffset = wasm._malloc(paramsByteLength);

  // handles the dimensions-related createTensor parameters
  const dims = [parametersSize];

  const dimsOffset = wasm.stackAlloc(4);
  const dimsIndex = dimsOffset / 4;
  wasm.HEAP32[dimsIndex] = parametersSize;

  try {
    // wraps allocated array in a tensor
    tensor = wasm._OrtCreateTensor(
        tensorDataTypeStringToEnum(tensorTypeAsString), paramsOffset, paramsByteLength, dimsOffset, dims.length,
        dataLocationStringToEnum(locationAsString));
    ifErrCodeCheckLastError(
        tensor, `Can't create tensor for getContiguousParameters. session=${trainingSessionId}.`, false);

    if (wasm._OrtTrainingCopyParametersToBuffer) {
      const errCode = wasm._OrtTrainingCopyParametersToBuffer(trainingSessionId, tensor, parametersSize, trainableOnly);
      ifErrCodeCheckLastError(errCode, 'Can\'t get contiguous parameters.');

    } else {
      throw new Error(NO_TRAIN_FUNCS_MSG);
    }

    // copies from WASM memory to a JavaScript typed array, which is then put into a TensorMetadata object
    const typedArrayConstructor = tensorTypeToTypedArrayConstructor(tensorTypeAsString);
    const data = new typedArrayConstructor(parametersSize);
    const output: TensorMetadata[] = [];
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
        .set(wasm.HEAPU8.subarray(paramsOffset, paramsOffset + paramsByteLength));
    output.push([tensorTypeAsString, dims, data, locationAsString]);
    if (output.length !== 1) {
      throw new Error(`something unexpected happened in the getContiguousParameters function. Expected output length of
     one, got ${output.length}`);
    } else {
      return output[0];
    }
  } finally {
    if (tensor !== 0) {
      wasm._OrtReleaseTensor(tensor);
    }
    wasm._free(paramsOffset);
    wasm._free(dimsOffset);
    wasm.stackRestore(stack);
  }
};

export const loadParametersBuffer =
    async(trainingSessionId: number, buffer: Uint8Array, trainableOnly: boolean): Promise<void> => {
  const wasm = getInstance();
  const stack = wasm.stackSave();

  const tensorTypeAsString = 'float32';
  const locationAsString = 'cpu';

  // allocates & copies JavaScript buffer to WASM heap
  const bufferByteLength = buffer.length;
  const bufferCount = bufferByteLength / 4;
  const bufferOffset = wasm._malloc(bufferByteLength);
  wasm.HEAPU8.set(buffer, bufferOffset);

  // allocates and handles moving dimensions information to WASM memory
  const dimsOffset = wasm.stackAlloc(4);
  wasm.HEAP32[dimsOffset / 4] = bufferCount;
  const dimsLength = 1;
  let tensor = 0;

  try {
    tensor = wasm._OrtCreateTensor(
        tensorDataTypeStringToEnum(tensorTypeAsString), bufferOffset, bufferByteLength, dimsOffset, dimsLength,
        dataLocationStringToEnum(locationAsString));
    ifErrCodeCheckLastError(tensor, `Can't create tensor for input/output. session=${trainingSessionId}`, false);

    if (wasm._OrtTrainingCopyParametersFromBuffer) {
      const errCode = wasm._OrtTrainingCopyParametersFromBuffer(trainingSessionId, tensor, bufferCount, trainableOnly);
      ifErrCodeCheckLastError(errCode, 'Can\'t copy buffer to parameters.');
    } else {
      throw new Error(NO_TRAIN_FUNCS_MSG);
    }
  } finally {
    if (tensor !== 0) {
      wasm._OrtReleaseTensor(tensor);
    }
    wasm.stackRestore(stack);
    wasm._free(bufferOffset);
    wasm._free(dimsOffset);
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
