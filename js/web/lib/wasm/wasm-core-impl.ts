// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Env, InferenceSession, Tensor} from 'onnxruntime-common';

import {init as initJsep} from './jsep/init';
import {SerializableModeldata, SerializableSessionMetadata, SerializableTensor} from './proxy-messages';
import {setRunOptions} from './run-options';
import {setSessionOptions} from './session-options';
import {allocWasmString} from './string-utils';
import {logLevelStringToEnum, tensorDataTypeEnumToString, tensorDataTypeStringToEnum, tensorTypeToTypedArrayConstructor} from './wasm-common';
import {getInstance} from './wasm-factory';

/**
 * initialize ORT environment.
 * @param numThreads SetGlobalIntraOpNumThreads(numThreads)
 * @param loggingLevel CreateEnv(static_cast<OrtLoggingLevel>(logging_level))
 */
const initOrt = async(numThreads: number, loggingLevel: number): Promise<void> => {
  const errorCode = getInstance()._OrtInit(numThreads, loggingLevel);
  if (errorCode !== 0) {
    throw new Error(`Can't initialize onnxruntime. error code = ${errorCode}`);
  }
};

/**
 * intialize runtime environment.
 * @param env passed in the environment config object.
 */
export const initRuntime = async(env: Env): Promise<void> => {
  // init ORT
  await initOrt(env.wasm.numThreads!, logLevelStringToEnum(env.logLevel));

  // init JSEP if available
  await initJsep(getInstance(), env);
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
export const createSessionAllocate = async (model: Uint8Array | { reader: ReadableStreamDefaultReader<Uint8Array>; size: number }): Promise<[number, number]> => {
  const wasm = getInstance();
  if (model instanceof Uint8Array) {
    const modelDataOffset = wasm._malloc(model.byteLength);
    wasm.HEAPU8.set(model, modelDataOffset);
    return [modelDataOffset, model.byteLength];
  } else {
    const modelDataOffset = wasm._malloc(model.size) >>> 0;
    let offset = 0;
    while (true) {
      const { done, value } = await model.reader.read();

      if (done) {
        break;
      }

      const memory = new Uint8Array(wasm.HEAPU8.buffer, modelDataOffset + offset, value.byteLength);
      memory.set(new Uint8Array(value.buffer));

      offset += value.byteLength;
      // console.log('wrote chunk', offset);
    }
    return [modelDataOffset, model.size];
  }
};

export const createSessionFinalize =
    (modelData: SerializableModeldata, options?: InferenceSession.SessionOptions): SerializableSessionMetadata => {
      const wasm = getInstance();

      let sessionHandle = 0;
      let sessionOptionsHandle = 0;
      let allocs: number[] = [];

      try {
        [sessionOptionsHandle, allocs] = setSessionOptions(options);

        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        sessionHandle = BigInt(wasm._OrtCreateSession(BigInt(modelData[0]), BigInt(modelData[1]),
            BigInt(sessionOptionsHandle)));
        if (sessionHandle === 0) {
          throw new Error('Can\'t create a session');
        }
      } finally {
        wasm._free(modelData[0]);
        console.log('FREED MEMORY');
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
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        const name = wasm._OrtGetInputName(sessionHandle, BigInt(i));
        if (name === 0) {
          throw new Error('Can\'t get an input name');
        }
        inputNamesUTF8Encoded.push(name);
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        inputNames.push(wasm.UTF8ToString(BigInt(name)));
      }
      console.log('inputs', inputNames, inputNamesUTF8Encoded)
      for (let i = 0; i < outputCount; i++) {
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        const name = wasm._OrtGetOutputName(sessionHandle, BigInt(i));
        if (name === 0) {
          throw new Error('Can\'t get an output name');
        }
        outputNamesUTF8Encoded.push(name);
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        outputNames.push(wasm.UTF8ToString(BigInt(name)));
      }

      activeSessions.set(sessionHandle, [sessionHandle, inputNamesUTF8Encoded, outputNamesUTF8Encoded]);
      return [sessionHandle, inputNames, outputNames];
    };


/**
 * create an instance of InferenceSession.
 * @returns the metadata of InferenceSession. 0-value handle for failure.
 */
export const createSession =
    async (model: Uint8Array, options?: InferenceSession.SessionOptions): Promise<SerializableSessionMetadata> => {
      const modelData: SerializableModeldata = await createSessionAllocate(model);
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

  const inputValues: unknown[] = [];
  const inputAllocs: unknown[] = [];

  try {
    [runOptionsHandle, runOptionsAllocs] = setRunOptions(options);

    // create input tensors
    for (let i = 0; i < inputCount; i++) {
      const dataType = inputs[i][0];
      const dims = inputs[i][1];
      const data = inputs[i][2];

      let dataOffset: bigint;
      let dataByteLength: number;

      if (Array.isArray(data)) {
        // string tensor
        dataByteLength = 4 * data.length;
        dataOffset = BigInt(wasm._malloc(dataByteLength));
        inputAllocs.push(BigInt(dataOffset));
        let dataIndex = Number(dataOffset / BigInt(4));
        for (let i = 0; i < data.length; i++) {
          if (typeof data[i] !== 'string') {
            throw new TypeError(`tensor data at index ${i} is not a string`);
          }
          // @ts-ignore
          wasm.HEAPU32[dataIndex++] = allocWasmString(data[i], inputAllocs);
        }
      } else {
        dataByteLength = data.byteLength;
        // @ts-ignore
        dataOffset = wasm._malloc(dataByteLength);
        inputAllocs.push(dataOffset);

        // console.log('SETTING INPUT OFFSET', dataOffset, dataOffset >>> 0);
        // eslint-disable-next-line no-bitwise
        // @ts-ignore
        wasm.HEAPU8.set(new Uint8Array(data.buffer, data.byteOffset, dataByteLength), dataOffset >>> 0);
        // console.log('SET INPUT OFFSET');
      }

      const stack = wasm.stackSave();
      const dimsOffset = BigInt(wasm.stackAlloc(8 * dims.length));
      try {
        console.log('creating tensor', tensorDataTypeStringToEnum(dataType), dataOffset, dataByteLength,
            dimsOffset, dims.length);
        dims.forEach((d, idx) => wasm.HEAPU64[Number(dimsOffset / BigInt(8) + BigInt(idx))] = BigInt(d));
        const tensor = wasm._OrtCreateTensor(
            // eslint-disable-next-line no-bitwise
            // @ts-ignore
            tensorDataTypeStringToEnum(dataType), BigInt(dataOffset), BigInt(dataByteLength), BigInt(dimsOffset),
            BigInt(dims.length));
        if (tensor === 0) {
          throw new Error('Can\'t create a tensor');
        }
        inputValues.push(tensor);
        console.log('created tensor');
      } finally {
        wasm.stackRestore(stack);
      }
    }

    const beforeRunStack = wasm.stackSave();
    const inputValuesOffset = BigInt(wasm.stackAlloc(inputCount * 8));
    const inputNamesOffset = BigInt(wasm.stackAlloc(inputCount * 8));
    const outputValuesOffset = BigInt(wasm.stackAlloc(outputCount * 8));
    const outputNamesOffset = BigInt(wasm.stackAlloc(outputCount * 8));

    try {
      let inputValuesIndex = Number(inputValuesOffset / BigInt(8));
      let inputNamesIndex = Number(inputNamesOffset / BigInt(8));
      let outputValuesIndex = Number(outputValuesOffset / BigInt(8));
      let outputNamesIndex = Number(outputNamesOffset / BigInt(8));
      for (let i = 0; i < inputCount; i++) {
        // @ts-ignore
        wasm.HEAPU64[inputValuesIndex++] = inputValues[i];
        wasm.HEAPU64[inputNamesIndex++] = BigInt(inputNamesUTF8Encoded[inputIndices[i]]);
      }
      for (let i = 0; i < outputCount; i++) {
        wasm.HEAPU64[outputValuesIndex++] = BigInt(0);
        wasm.HEAPU64[outputNamesIndex++] = BigInt(outputNamesUTF8Encoded[outputIndices[i]]);
      }

      console.log('running', sessionHandle, inputNamesOffset, inputValuesOffset, inputCount, outputNamesOffset, outputCount,
          outputValuesOffset, runOptionsHandle);
      // support RunOptions
      let errorCode = wasm._OrtRun(
          // @ts-ignore
          sessionHandle, inputNamesOffset, inputValuesOffset, BigInt(inputCount), outputNamesOffset, BigInt(outputCount),
          outputValuesOffset, runOptionsHandle);
      console.log('run', errorCode);

      // eslint-disable-next-line @typescript-eslint/naming-convention
      const runPromise = wasm.jsepRunPromise;
      if (runPromise && typeof runPromise.then !== 'undefined') {
        errorCode = await runPromise;
      }

      const output: SerializableTensor[] = [];

      if (errorCode === 0) {
        for (let i = 0; i < outputCount; i++) {
          const tensor = wasm.HEAPU64[Number(outputValuesOffset / BigInt(8)) + i];

          const beforeGetTensorDataStack = wasm.stackSave();
          // stack allocate 4 pointer value
          const tensorDataOffset = wasm.stackAlloc(4 * 8);

          let type: Tensor.Type|undefined, dataOffset = 0;
          try {
            errorCode = wasm._OrtGetTensorData(
                // @ts-ignore
                tensor, BigInt(tensorDataOffset), BigInt(tensorDataOffset + 8),
                BigInt(tensorDataOffset + 16), BigInt(tensorDataOffset + 24));
            if (errorCode !== 0) {
              throw new Error(`Can't access output tensor data. error code = ${errorCode}`);
            }
            let tensorDataIndex = tensorDataOffset / 8;
            const dataType = wasm.HEAPU64[tensorDataIndex++];
            // @ts-ignore
            dataOffset = wasm.HEAPU64[tensorDataIndex++];
            const dimsOffset = wasm.HEAPU64[tensorDataIndex++];
            const dimsLength = wasm.HEAPU64[tensorDataIndex++];
            const dims = [];
            for (let i = 0; i < dimsLength; i++) {
              dims.push(wasm.HEAPU64[Number(dimsOffset / BigInt(8)) + i]);
            }
            // @ts-ignore
            wasm._OrtFree(dimsOffset);

            const size = dims.length === 0 ? BigInt(1) : dims.reduce((a, b) => a * b);
            type = tensorDataTypeEnumToString(Number(dataType));
            if (type === 'string') {
              const stringData: string[] = [];
              let dataIndex = dataOffset / 4;
              for (let i = 0; i < size; i++) {
                const offset = wasm.HEAPU32[dataIndex++];
                const maxBytesToRead = BigInt(i) === size - BigInt(1) ? undefined : wasm.HEAPU32[dataIndex] - offset;
                stringData.push(wasm.UTF8ToString(offset, maxBytesToRead));
              }
              // @ts-ignore
              output.push([type, dims, stringData]);
            } else {
              const typedArrayConstructor = tensorTypeToTypedArrayConstructor(type);
              const data = new typedArrayConstructor(Number(size));
              new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
                  .set(wasm.HEAPU8.subarray(dataOffset, dataOffset + data.byteLength));
              // @ts-ignore
              output.push([type, dims, data]);
            }
          } finally {
            wasm.stackRestore(beforeGetTensorDataStack);
            if (type === 'string' && dataOffset) {
              // @ts-ignore
              wasm._free(dataOffset);
            }
            // @ts-ignore
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
    // @ts-ignore
    inputValues.forEach(wasm._OrtReleaseTensor);
    // @ts-ignore
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
