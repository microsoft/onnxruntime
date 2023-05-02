// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession, Tensor} from 'onnxruntime-common';

import {SerializableModeldata, SerializableSessionMetadata, SerializableTensor} from './proxy-messages';
import {setRunOptions} from './run-options';
import {setSessionOptions} from './session-options';
import {allocWasmString} from './string-utils';
import {tensorDataTypeEnumToString, tensorDataTypeStringToEnum, tensorTypeToTypedArrayConstructor} from './wasm-common';
import {getInstance} from './wasm-factory';

/**
 * initialize ORT environment.
 * @param numThreads SetGlobalIntraOpNumThreads(numThreads)
 * @param loggingLevel CreateEnv(static_cast<OrtLoggingLevel>(logging_level))
 */
export const initOrt = (numThreads: number, loggingLevel: number): void => {
  const errorCode = getInstance()._OrtInit(numThreads, loggingLevel);
  if (errorCode !== 0) {
    throw new Error(`Can't initialize onnxruntime. error code = ${errorCode}`);
  }
};

/**
 *  tuple elements are: InferenceSession ID; inputNamesUTF8Encoded; outputNamesUTF8Encoded
 */
type SessionMetadata = [number, number[], number[]];

const activeSessions = new Map<number, SessionMetadata>();

/**
 * create an instance of InferenceSession.
 * @returns the metadata of InferenceSession. 0-value handle for failure.
 */
export const createSessionAllocate = (model: Uint8Array): [number, number] => {
  const wasm = getInstance();
  const modelDataOffset = wasm._malloc(model.byteLength);
  wasm.HEAPU8.set(model, modelDataOffset);
  return [modelDataOffset, model.byteLength];
};

export const createSessionFinalize =
    (modelData: SerializableModeldata, options?: InferenceSession.SessionOptions): SerializableSessionMetadata => {
      const wasm = getInstance();

      let sessionHandle = 0;
      let sessionOptionsHandle = 0;
      let allocs: number[] = [];

      try {
        [sessionOptionsHandle, allocs] = setSessionOptions(options);

        sessionHandle = wasm._OrtCreateSession(modelData[0], modelData[1], sessionOptionsHandle);
        if (sessionHandle === 0) {
          throw new Error('Can\'t create a session');
        }
      } finally {
        wasm._free(modelData[0]);
        if (sessionOptionsHandle !== 0) {
          wasm._OrtReleaseSessionOptions(sessionOptionsHandle);
        }
        allocs.forEach(wasm._free);
      }

      const inputCount = wasm._OrtGetInputCount(sessionHandle);
      const outputCount = wasm._OrtGetOutputCount(sessionHandle);

      const inputNames = [];
      const inputNamesUTF8Encoded = [];
      const outputNames = [];
      const outputNamesUTF8Encoded = [];
      for (let i = 0; i < inputCount; i++) {
        const name = wasm._OrtGetInputName(sessionHandle, i);
        if (name === 0) {
          throw new Error('Can\'t get an input name');
        }
        inputNamesUTF8Encoded.push(name);
        inputNames.push(wasm.UTF8ToString(name));
      }
      for (let i = 0; i < outputCount; i++) {
        const name = wasm._OrtGetOutputName(sessionHandle, i);
        if (name === 0) {
          throw new Error('Can\'t get an output name');
        }
        outputNamesUTF8Encoded.push(name);
        outputNames.push(wasm.UTF8ToString(name));
      }

      activeSessions.set(sessionHandle, [sessionHandle, inputNamesUTF8Encoded, outputNamesUTF8Encoded]);
      return [sessionHandle, inputNames, outputNames];
    };


/**
 * create an instance of InferenceSession.
 * @returns the metadata of InferenceSession. 0-value handle for failure.
 */
export const createSession =
    (model: Uint8Array, options?: InferenceSession.SessionOptions): SerializableSessionMetadata => {
      const modelData: SerializableModeldata = createSessionAllocate(model);
      return createSessionFinalize(modelData, options);
    };

export const releaseSession = (sessionId: number): void => {
  const wasm = getInstance();
  const session = activeSessions.get(sessionId);
  if (!session) {
    throw new Error('invalid session id');
  }
  const sessionHandle = session[0];
  const inputNamesUTF8Encoded = session[1];
  const outputNamesUTF8Encoded = session[2];

  inputNamesUTF8Encoded.forEach(wasm._OrtFree);
  outputNamesUTF8Encoded.forEach(wasm._OrtFree);
  wasm._OrtReleaseSession(sessionHandle);
  activeSessions.delete(sessionId);
};

/**
 * perform inference run
 */
export const run = async(
    sessionId: number, inputIndices: number[], inputs: SerializableTensor[], outputIndices: number[],
    options: InferenceSession.RunOptions): Promise<SerializableTensor[]> => {
  const wasm = getInstance();
  const session = activeSessions.get(sessionId);
  if (!session) {
    throw new Error('invalid session id');
  }
  const sessionHandle = session[0];
  const inputNamesUTF8Encoded = session[1];
  const outputNamesUTF8Encoded = session[2];

  const inputCount = inputIndices.length;
  const outputCount = outputIndices.length;

  let runOptionsHandle = 0;
  let runOptionsAllocs: number[] = [];

  const inputValues: number[] = [];
  const inputAllocs: number[] = [];

  try {
    [runOptionsHandle, runOptionsAllocs] = setRunOptions(options);

    // create input tensors
    for (let i = 0; i < inputCount; i++) {
      const dataType = inputs[i][0];
      const dims = inputs[i][1];
      const data = inputs[i][2];

      let dataOffset: number;
      let dataByteLength: number;

      if (Array.isArray(data)) {
        // string tensor
        dataByteLength = 4 * data.length;
        dataOffset = wasm._malloc(dataByteLength);
        inputAllocs.push(dataOffset);
        let dataIndex = dataOffset / 4;
        for (let i = 0; i < data.length; i++) {
          if (typeof data[i] !== 'string') {
            throw new TypeError(`tensor data at index ${i} is not a string`);
          }
          wasm.HEAPU32[dataIndex++] = allocWasmString(data[i], inputAllocs);
        }
      } else {
        dataByteLength = data.byteLength;
        dataOffset = wasm._malloc(dataByteLength);
        inputAllocs.push(dataOffset);
        wasm.HEAPU8.set(new Uint8Array(data.buffer, data.byteOffset, dataByteLength), dataOffset);
      }

      const stack = wasm.stackSave();
      const dimsOffset = wasm.stackAlloc(4 * dims.length);
      try {
        let dimIndex = dimsOffset / 4;
        dims.forEach(d => wasm.HEAP32[dimIndex++] = d);
        const tensor = wasm._OrtCreateTensor(
            tensorDataTypeStringToEnum(dataType), dataOffset, dataByteLength, dimsOffset, dims.length);
        if (tensor === 0) {
          throw new Error('Can\'t create a tensor');
        }
        inputValues.push(tensor);
      } finally {
        wasm.stackRestore(stack);
      }
    }

    const beforeRunStack = wasm.stackSave();
    const inputValuesOffset = wasm.stackAlloc(inputCount * 4);
    const inputNamesOffset = wasm.stackAlloc(inputCount * 4);
    const outputValuesOffset = wasm.stackAlloc(outputCount * 4);
    const outputNamesOffset = wasm.stackAlloc(outputCount * 4);

    try {
      let inputValuesIndex = inputValuesOffset / 4;
      let inputNamesIndex = inputNamesOffset / 4;
      let outputValuesIndex = outputValuesOffset / 4;
      let outputNamesIndex = outputNamesOffset / 4;
      for (let i = 0; i < inputCount; i++) {
        wasm.HEAPU32[inputValuesIndex++] = inputValues[i];
        wasm.HEAPU32[inputNamesIndex++] = inputNamesUTF8Encoded[inputIndices[i]];
      }
      for (let i = 0; i < outputCount; i++) {
        wasm.HEAPU32[outputValuesIndex++] = 0;
        wasm.HEAPU32[outputNamesIndex++] = outputNamesUTF8Encoded[outputIndices[i]];
      }

      // support RunOptions
      let errorCode = wasm._OrtRun(
          sessionHandle, inputNamesOffset, inputValuesOffset, inputCount, outputNamesOffset, outputCount,
          outputValuesOffset, runOptionsHandle);

      // eslint-disable-next-line @typescript-eslint/naming-convention
      const runPromise = wasm.jsepRunPromise;
      if (runPromise && typeof runPromise.then !== 'undefined') {
        errorCode = await runPromise;
      }

      const output: SerializableTensor[] = [];

      if (errorCode === 0) {
        for (let i = 0; i < outputCount; i++) {
          const tensor = wasm.HEAPU32[outputValuesOffset / 4 + i];

          const beforeGetTensorDataStack = wasm.stackSave();
          // stack allocate 4 pointer value
          const tensorDataOffset = wasm.stackAlloc(4 * 4);

          let type: Tensor.Type|undefined, dataOffset = 0;
          try {
            errorCode = wasm._OrtGetTensorData(
                tensor, tensorDataOffset, tensorDataOffset + 4, tensorDataOffset + 8, tensorDataOffset + 12);
            if (errorCode !== 0) {
              throw new Error(`Can't access output tensor data. error code = ${errorCode}`);
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

            const size = dims.length === 0 ? 1 : dims.reduce((a, b) => a * b);
            type = tensorDataTypeEnumToString(dataType);
            if (type === 'string') {
              const stringData: string[] = [];
              let dataIndex = dataOffset / 4;
              for (let i = 0; i < size; i++) {
                const offset = wasm.HEAPU32[dataIndex++];
                const maxBytesToRead = i === size - 1 ? undefined : wasm.HEAPU32[dataIndex] - offset;
                stringData.push(wasm.UTF8ToString(offset, maxBytesToRead));
              }
              output.push([type, dims, stringData]);
            } else {
              const typedArrayConstructor = tensorTypeToTypedArrayConstructor(type);
              const data = new typedArrayConstructor(size);
              new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
                  .set(wasm.HEAPU8.subarray(dataOffset, dataOffset + data.byteLength));
              output.push([type, dims, data]);
            }
          } finally {
            wasm.stackRestore(beforeGetTensorDataStack);
            if (type === 'string' && dataOffset) {
              wasm._free(dataOffset);
            }
            wasm._OrtReleaseTensor(tensor);
          }
        }
      }

      if (errorCode === 0) {
        return output;
      } else {
        throw new Error(`failed to call OrtRun(). error code = ${errorCode}.`);
      }
    } finally {
      wasm.stackRestore(beforeRunStack);
    }
  } finally {
    inputValues.forEach(wasm._OrtReleaseTensor);
    inputAllocs.forEach(wasm._free);

    wasm._OrtReleaseRunOptions(runOptionsHandle);
    runOptionsAllocs.forEach(wasm._free);
  }
};

/**
 * end profiling
 */
export const endProfiling = (sessionId: number): void => {
  const wasm = getInstance();
  const session = activeSessions.get(sessionId);
  if (!session) {
    throw new Error('invalid session id');
  }
  const sessionHandle = session[0];

  // profile file name is not used yet, but it must be freed.
  const profileFileName = wasm._OrtEndProfiling(sessionHandle);
  if (profileFileName === 0) {
    throw new Error('Can\'t get an profile file name');
  }
  wasm._OrtFree(profileFileName);
};

export const extractTransferableBuffers = (tensors: readonly SerializableTensor[]): ArrayBufferLike[] => {
  const buffers: ArrayBufferLike[] = [];
  for (const tensor of tensors) {
    const data = tensor[2];
    if (!Array.isArray(data) && data.buffer) {
      buffers.push(data.buffer);
    }
  }
  return buffers;
};
