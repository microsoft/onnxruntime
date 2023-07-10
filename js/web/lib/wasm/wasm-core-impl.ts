// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Env, InferenceSession, Tensor} from 'onnxruntime-common';

import {init as initJsep} from './jsep/init';
import {SerializableModeldata, SerializableSessionMetadata, SerializableTensor} from './proxy-messages';
import {setRunOptions} from './run-options';
import {setSessionOptions} from './session-options';
import {logLevelStringToEnum, tensorDataTypeEnumToString, tensorDataTypeStringToEnum, tensorTypeToTypedArrayConstructor} from './wasm-common';
import {getInstance} from './wasm-factory';
import {allocWasmString, checkLastError} from './wasm-utils';

/**
 * get the input/output count of the session.
 * @param sessionHandle the handle representing the session. should be non-zero.
 * @returns a tuple including 2 numbers, representing the input count and output count.
 */
const getSessionInputOutputCount = (sessionHandle: number): [number, number] => {
  const wasm = getInstance();
  const stack = wasm.stackSave();
  try {
    const dataOffset = wasm.stackAlloc(8);
    const errorCode = wasm._OrtGetInputOutputCount(sessionHandle, dataOffset, dataOffset + 4);
    if (errorCode !== 0) {
      checkLastError('Can\'t get session input/output count.');
    }
    return [wasm.HEAP32[dataOffset / 4], wasm.HEAP32[dataOffset / 4 + 1]];
  } finally {
    wasm.stackRestore(stack);
  }
};

/**
 * initialize ORT environment.
 * @param numThreads SetGlobalIntraOpNumThreads(numThreads)
 * @param loggingLevel CreateEnv(static_cast<OrtLoggingLevel>(logging_level))
 */
const initOrt = async(numThreads: number, loggingLevel: number): Promise<void> => {
  const errorCode = getInstance()._OrtInit(numThreads, loggingLevel);
  if (errorCode !== 0) {
    checkLastError('Can\'t initialize onnxruntime.');
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
type SessionMetadata = [bigint, bigint[], bigint[]];

const activeSessions = new Map<bigint, SessionMetadata>();

/**
 * allocate the memory and memcpy the model bytes, preparing for creating an instance of InferenceSession.
 * @returns a 2-elements tuple - the pointer and size of the allocated buffer
 */
export const createSessionAllocate = async (model: Uint8Array | { reader: ReadableStreamDefaultReader<Uint8Array>; size: number }): Promise<[bigint, number]> => {
  const wasm = getInstance();
  if (model instanceof Uint8Array) {
    const modelDataOffset = wasm._malloc(BigInt(model.byteLength));
    // wasm.HEAPU8.set(model, modelDataOffset);
    return [modelDataOffset, model.byteLength];
  } else {
    const modelDataOffset = wasm._malloc(BigInt(model.size));
    let offset = 0;
    while (true) {
      const { done, value } = await model.reader.read();

      if (done) {
        break;
      }

      const memory = new Uint8Array(wasm.HEAPU8.buffer, Number(modelDataOffset) + offset, value.byteLength);
      memory.set(new Uint8Array(value.buffer));

      offset += value.byteLength;
      // console.log('wrote chunk', offset);
    }
    return [modelDataOffset, model.size];
  }
};

/**
 * create an inference session using the prepared buffer containing the model data.
 * @param modelData a 2-elements tuple containing the pointer and size of the model data buffer.
 * @param options an optional session options object.
 * @returns a 3-elements tuple containing [session handle, input names, output names]
 */
export const createSessionFinalize =
    (modelData: SerializableModeldata, options?: InferenceSession.SessionOptions): SerializableSessionMetadata => {
      const wasm = getInstance();

      let sessionHandle = BigInt(0);
      let sessionOptionsHandle = BigInt(0);
      let allocs: bigint[] = [];
      const inputNamesUTF8Encoded = [];
      const outputNamesUTF8Encoded = [];

      try {
        [sessionOptionsHandle, allocs] = setSessionOptions(options);

        sessionHandle = wasm._OrtCreateSession(modelData[0], modelData[1], sessionOptionsHandle);
        if (sessionHandle === 0) {
          checkLastError('Can\'t create a session.');
        }

        const [inputCount, outputCount] = getSessionInputOutputCount(sessionHandle);

        const inputNames = [];
        const outputNames = [];
        for (let i = 0; i < inputCount; i++) {
          // eslint-disable-next-line @typescript-eslint/ban-ts-comment
          // @ts-ignore
          const name = wasm._OrtGetInputName(sessionHandle, BigInt(i));
          if (name === BigInt(0)) {
            throw new Error('Can\'t get an input name');
          }
          inputNamesUTF8Encoded.push(name);
          // eslint-disable-next-line @typescript-eslint/ban-ts-comment
          // @ts-ignore
          inputNames.push(wasm.UTF8ToString(Number(name)));
        }
        console.log('inputs', inputNames, inputNamesUTF8Encoded);
        for (let i = 0; i < outputCount; i++) {
          // eslint-disable-next-line @typescript-eslint/ban-ts-comment
          // @ts-ignore
          const name = wasm._OrtGetOutputName(sessionHandle, BigInt(i));
          if (name === BigInt(0)) {
            throw new Error('Can\'t get an output name');
          }
          outputNamesUTF8Encoded.push(name);
          // eslint-disable-next-line @typescript-eslint/ban-ts-comment
          // @ts-ignore
          outputNames.push(wasm.UTF8ToString(Number(name)));
        }

        activeSessions.set(sessionHandle, [sessionHandle, inputNamesUTF8Encoded, outputNamesUTF8Encoded]);
        return [sessionHandle, inputNames, outputNames];
      } catch (e) {
        inputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));
        outputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));

        if (sessionHandle !== 0) {
          wasm._OrtReleaseSession(sessionHandle);
        }
        throw e;
      } finally {
        wasm._free(modelData[0]);
        if (sessionOptionsHandle !== 0) {
          wasm._OrtReleaseSessionOptions(sessionOptionsHandle);
        }
        allocs.forEach(alloc => wasm._free(alloc));
      }
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

export const releaseSession = (sessionId: bigint): void => {
  const wasm = getInstance();
  const session = activeSessions.get(sessionId);
  if (!session) {
    throw new Error(`cannot release session. invalid session id: ${sessionId}`);
  }
  const [sessionHandle, inputNamesUTF8Encoded, outputNamesUTF8Encoded] = session;

  inputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));
  outputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));
  wasm._OrtReleaseSession(sessionHandle);
  activeSessions.delete(sessionId);
};

/**
 * perform inference run
 */
export const run = async(
    sessionId: bigint, inputIndices: number[], inputs: SerializableTensor[], outputIndices: number[],
    options: InferenceSession.RunOptions): Promise<SerializableTensor[]> => {
  const wasm = getInstance();
  const session = activeSessions.get(sessionId);
  if (!session) {
    throw new Error(`cannot run inference. invalid session id: ${sessionId}`);
  }
  const [sessionHandle, inputNamesUTF8Encoded, outputNamesUTF8Encoded] = session;

  const inputCount = inputIndices.length;
  const outputCount = outputIndices.length;

  let runOptionsHandle = BigInt(0);
  let runOptionsAllocs: bigint[] = [];

  const inputValues: bigint[] = [];
  const inputAllocs: bigint[] = [];

  try {
    [runOptionsHandle, runOptionsAllocs] = setRunOptions(options);

    // create input tensors
    for (let i = 0; i < inputCount; i++) {
      const dataType = inputs[i][0];
      const dims = inputs[i][1];
      const data = inputs[i][2];

      let dataOffset: bigint;
      let dataByteLength: bigint;

      if (Array.isArray(data)) {
        // string tensor
        dataByteLength = BigInt(4 * data.length);
        dataOffset = wasm._malloc(dataByteLength);
        inputAllocs.push(BigInt(dataOffset));
        let dataIndex = Number(dataOffset) / 4;
        for (let i = 0; i < data.length; i++) {
          if (typeof data[i] !== 'string') {
            throw new TypeError(`tensor data at index ${i} is not a string`);
          }
          wasm.HEAPU32[dataIndex++] = Number(allocWasmString(data[i], inputAllocs));
        }
      } else {
        dataByteLength = BigInt(data.byteLength);
        dataOffset = wasm._malloc(dataByteLength);
        inputAllocs.push(dataOffset);
        wasm.HEAPU8.set(new Uint8Array(data.buffer, data.byteOffset, Number(dataByteLength)), Number(dataOffset));
      }

      // const stack = wasm.stackSave();
      const dimsOffset = wasm._malloc(BigInt(8 * dims.length));
      try {
        let dimIndex = Number(dimsOffset) / 8;
        dims.forEach(d => wasm.HEAPU64[dimIndex++] = BigInt(d));
        const tensor = wasm._OrtCreateTensor(
            tensorDataTypeStringToEnum(dataType), BigInt(dataOffset), BigInt(dataByteLength), BigInt(dimsOffset),
            BigInt(dims.length));
        if (tensor === BigInt(0)) {
          checkLastError(`Can't create tensor for input[${i}].`);
        }
        inputValues.push(tensor);
      } finally {
        // wasm.stackRestore(stack);
      }
    }

    // const beforeRunStack = wasm.stackSave();
    const inputValuesOffset = BigInt(wasm._malloc(BigInt(inputCount * 8)));
    const inputNamesOffset = BigInt(wasm._malloc(BigInt(inputCount * 8)));
    const outputValuesOffset = BigInt(wasm._malloc(BigInt(outputCount * 8)));
    const outputNamesOffset = BigInt(wasm._malloc(BigInt(outputCount * 8)));

    try {
      let inputValuesIndex = Number(inputValuesOffset / BigInt(8));
      let inputNamesIndex = Number(inputNamesOffset / BigInt(8));
      let outputValuesIndex = Number(outputValuesOffset / BigInt(8));
      let outputNamesIndex = Number(outputNamesOffset / BigInt(8));
      for (let i = 0; i < inputCount; i++) {
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
          sessionHandle, BigInt(inputNamesOffset), BigInt(inputValuesOffset), BigInt(inputCount),
          BigInt(outputNamesOffset), BigInt(outputCount),
          BigInt(outputValuesOffset), runOptionsHandle);
      console.log('run', errorCode);

      // eslint-disable-next-line @typescript-eslint/naming-convention
      const runPromise = wasm.jsepRunPromise;
      if (runPromise && typeof runPromise.then !== 'undefined') {
        errorCode = await runPromise;
      }

      const output: SerializableTensor[] = [];

      if (errorCode !== 0) {
        checkLastError('failed to call OrtRun().');
      }

      for (let i = 0; i < outputCount; i++) {
        const tensor = wasm.HEAPU64[Number(outputValuesOffset / BigInt(8)) + i];

        // const beforeGetTensorDataStack = wasm.stackSave();
          // stack allocate 4 pointer value
          const tensorDataOffset = wasm._malloc(BigInt(4 * 8));

          let type: Tensor.Type|undefined, dataOffset = BigInt(0);
          try {
            console.log('get tensor data', tensorDataOffset)
            errorCode = wasm._OrtGetTensorData(
                // @ts-ignore
                tensor, BigInt(tensorDataOffset), BigInt(tensorDataOffset + 8),
                // @ts-ignore
                BigInt(tensorDataOffset + 16), BigInt(tensorDataOffset + 24));
            if (errorCode) {
              throw new Error(`Can't access output tensor data. error code = ${errorCode}`);
            }
            let tensorDataIndex = Number(tensorDataOffset) / 8;
            const dataType = wasm.HEAPU64[tensorDataIndex++];

            dataOffset = wasm.HEAPU64[tensorDataIndex++];
            const dimsOffset = wasm.HEAPU64[tensorDataIndex++];
            const dimsLength = wasm.HEAPU64[tensorDataIndex++];
            const dims = [];
            for (let i = 0; i < dimsLength; i++) {
              dims.push(wasm.HEAPU64[Number(dimsOffset / BigInt(8)) + i]);
            }

            wasm._OrtFree(dimsOffset);

            const size = dims.length === 0 ? BigInt(1) : dims.reduce((a, b) => a * b);
            type = tensorDataTypeEnumToString(Number(dataType));
            if (type === 'string') {
              console.log('STRING TENSOR')
              const stringData: string[] = [];
              let dataIndex = Number(dataOffset) / 4;
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
                  // @ts-ignore
                  .set(wasm.HEAPU8.subarray(Number(dataOffset), Number(dataOffset) + data.byteLength));
              // @ts-ignore
              output.push([type, dims, data]);
            }
          } finally {
            // wasm.stackRestore(beforeGetTensorDataStack);
            if (type === 'string' && dataOffset) {
              wasm._free(dataOffset);
            }
            wasm._OrtReleaseTensor(tensor);
          }
        }

      return output;
    } finally {
      wasm.stackRestore(beforeRunStack);
    }
  } finally {
    inputValues.forEach(v => wasm._OrtReleaseTensor(v));
    inputAllocs.forEach(p => wasm._free(p));

    if (runOptionsHandle !== 0) {
      wasm._OrtReleaseRunOptions(runOptionsHandle);
    }
    runOptionsAllocs.forEach(p => wasm._free(p));
  }
};

/**
 * end profiling
 */
export const endProfiling = (sessionId: bigint): void => {
  const wasm = getInstance();
  const session = activeSessions.get(sessionId);
  if (!session) {
    throw new Error('invalid session id');
  }
  const sessionHandle = session[0];

  // profile file name is not used yet, but it must be freed.
  const profileFileName = wasm._OrtEndProfiling(sessionHandle);
  if (profileFileName === BigInt(0)) {
    checkLastError('Can\'t get an profile file name.');
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
