// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Env, InferenceSession, Tensor} from 'onnxruntime-common';

import {SerializableModeldata, SerializableSessionMetadata, SerializableTensorMetadata, TensorMetadata} from './proxy-messages';
import {setRunOptions} from './run-options';
import {setSessionOptions} from './session-options';
import {dataLocationStringToEnum, getTensorElementSize, isGpuBufferSupportedType, logLevelStringToEnum, tensorDataTypeEnumToString, tensorDataTypeStringToEnum, tensorTypeToTypedArrayConstructor} from './wasm-common';
import {getInstance} from './wasm-factory';
import {allocWasmString, checkLastError} from './wasm-utils';

let ortEnvInitialized = false;

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
const initOrt = (numThreads: number, loggingLevel: number): void => {
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
  initOrt(env.wasm.numThreads!, logLevelStringToEnum(env.logLevel));

  if (!BUILD_DEFS.DISABLE_WEBGPU) {
    // init JSEP if available

    // eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-var-requires
    const initJsep = require('./jsep/init').init;
    await initJsep(getInstance(), env);
  }

  ortEnvInitialized = true;
};

/**
 * valid data locations for input/output tensors.
 */
type SupportedTensorDataLocationForInputOutput = 'cpu'|'cpu-pinned'|'gpu-buffer';

type IOBindingState = {
  /**
   * the handle of IO binding.
   */
  readonly handle: number;

  /**
   * the preferred location for each output tensor.
   *
   * value is one of 'cpu', 'cpu-pinned', 'gpu-buffer'.
   */
  readonly outputPreferredLocations: readonly SupportedTensorDataLocationForInputOutput[];

  /**
   * enum value of the preferred location for each output tensor.
   */
  readonly outputPreferredLocationsEncoded: readonly number[];
};

/**
 *  tuple elements are: InferenceSession ID; inputNamesUTF8Encoded; outputNamesUTF8Encoded; bindingState
 */
type SessionMetadata = [
  inferenceSessionId: number, inputNamesUTF8Encoded: number[], outputNamesUTF8Encoded: number[],
  bindingState: IOBindingState|null
];

const activeSessions = new Map<number, SessionMetadata>();

export const isOrtEnvInitialized = (): boolean => ortEnvInitialized;

/**
 * allocate the memory and memcpy the model bytes, preparing for creating an instance of InferenceSession.
 * @returns a 2-elements tuple - the pointer and size of the allocated buffer
 */
export const createSessionAllocate = (model: Uint8Array): [number, number] => {
  const wasm = getInstance();
  const modelDataOffset = wasm._malloc(model.byteLength);
  if (modelDataOffset === 0) {
    throw new Error(`Can't create a session. failed to allocate a buffer of size ${model.byteLength}.`);
  }
  wasm.HEAPU8.set(model, modelDataOffset);
  return [modelDataOffset, model.byteLength];
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

      let sessionHandle = 0;
      let sessionOptionsHandle = 0;
      let ioBindingHandle = 0;
      let allocs: number[] = [];
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
        const outputPreferredLocations: SupportedTensorDataLocationForInputOutput[] = [];
        for (let i = 0; i < inputCount; i++) {
          const name = wasm._OrtGetInputName(sessionHandle, i);
          if (name === 0) {
            checkLastError('Can\'t get an input name.');
          }
          inputNamesUTF8Encoded.push(name);
          inputNames.push(wasm.UTF8ToString(name));
        }
        for (let i = 0; i < outputCount; i++) {
          const name = wasm._OrtGetOutputName(sessionHandle, i);
          if (name === 0) {
            checkLastError('Can\'t get an output name.');
          }
          outputNamesUTF8Encoded.push(name);
          const nameString = wasm.UTF8ToString(name);
          outputNames.push(nameString);

          if (!BUILD_DEFS.DISABLE_WEBGPU) {
            const location = typeof options?.preferredOutputLocation === 'string' ?
                options.preferredOutputLocation :
                options?.preferredOutputLocation?.[nameString] ?? 'cpu';
            if (location !== 'cpu' && location !== 'cpu-pinned' && location !== 'gpu-buffer') {
              throw new Error(`Not supported preferred output location: ${location}.`);
            }
            outputPreferredLocations.push(location);
          }
        }

        // use IO binding only when at least one output is preffered to be on GPU.
        let bindingState: IOBindingState|null = null;
        if (!BUILD_DEFS.DISABLE_WEBGPU && outputPreferredLocations.some(l => l === 'gpu-buffer')) {
          ioBindingHandle = wasm._OrtCreateBinding(sessionHandle);
          if (ioBindingHandle === 0) {
            checkLastError('Can\'t create IO binding.');
          }

          bindingState = {
            handle: ioBindingHandle,
            outputPreferredLocations,
            outputPreferredLocationsEncoded: outputPreferredLocations.map(l => dataLocationStringToEnum(l)),
          };
        }

        activeSessions.set(sessionHandle, [sessionHandle, inputNamesUTF8Encoded, outputNamesUTF8Encoded, bindingState]);
        return [sessionHandle, inputNames, outputNames];
      } catch (e) {
        inputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));
        outputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));

        if (ioBindingHandle !== 0) {
          wasm._OrtReleaseBinding(ioBindingHandle);
        }

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
    (model: Uint8Array, options?: InferenceSession.SessionOptions): SerializableSessionMetadata => {
      const modelData: SerializableModeldata = createSessionAllocate(model);
      return createSessionFinalize(modelData, options);
    };

export const releaseSession = (sessionId: number): void => {
  const wasm = getInstance();
  const session = activeSessions.get(sessionId);
  if (!session) {
    throw new Error(`cannot release session. invalid session id: ${sessionId}`);
  }
  const [sessionHandle, inputNamesUTF8Encoded, outputNamesUTF8Encoded, ioBindingState] = session;

  if (ioBindingState) {
    wasm._OrtReleaseBinding(ioBindingState.handle);
  }

  wasm.jsepUnregisterBuffers?.(sessionId);

  inputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));
  outputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));
  wasm._OrtReleaseSession(sessionHandle);
  activeSessions.delete(sessionId);
};

const prepareInputOutputTensor =
    (tensor: TensorMetadata|null, tensorHandles: number[], allocs: number[], sessionId: number, index: number):
        void => {
          if (!tensor) {
            tensorHandles.push(0);
            return;
          }

          const wasm = getInstance();

          const dataType = tensor[0];
          const dims = tensor[1];
          const location = tensor[3];

          let rawData: number;
          let dataByteLength: number;

          if (dataType === 'string' && location === 'gpu-buffer') {
            throw new Error('String tensor is not supported on GPU.');
          }

          if (location === 'gpu-buffer') {
            const gpuBuffer = tensor[2].gpuBuffer as GPUBuffer;
            const elementSizeInBytes = getTensorElementSize(tensorDataTypeStringToEnum(dataType))!;
            dataByteLength = dims.reduce((a, b) => a * b, 1) * elementSizeInBytes;
            rawData = wasm.jsepRegisterBuffer(sessionId, index, gpuBuffer, dataByteLength);
          } else {
            const data = tensor[2];

            if (Array.isArray(data)) {
              // string tensor
              dataByteLength = 4 * data.length;
              rawData = wasm._malloc(dataByteLength);
              allocs.push(rawData);
              let dataIndex = rawData / 4;
              for (let i = 0; i < data.length; i++) {
                if (typeof data[i] !== 'string') {
                  throw new TypeError(`tensor data at index ${i} is not a string`);
                }
                wasm.HEAPU32[dataIndex++] = allocWasmString(data[i], allocs);
              }
            } else {
              dataByteLength = data.byteLength;
              rawData = wasm._malloc(dataByteLength);
              allocs.push(rawData);
              wasm.HEAPU8.set(new Uint8Array(data.buffer, data.byteOffset, dataByteLength), rawData);
            }
          }

          const stack = wasm.stackSave();
          const dimsOffset = wasm.stackAlloc(4 * dims.length);
          try {
            let dimIndex = dimsOffset / 4;
            dims.forEach(d => wasm.HEAP32[dimIndex++] = d);
            const tensor = wasm._OrtCreateTensor(
                tensorDataTypeStringToEnum(dataType), rawData, dataByteLength, dimsOffset, dims.length,
                dataLocationStringToEnum(location));
            if (tensor === 0) {
              checkLastError(`Can't create tensor for input/output. session=${sessionId}, index=${index}.`);
            }
            tensorHandles.push(tensor);
          } finally {
            wasm.stackRestore(stack);
          }
        };

/**
 * perform inference run
 */
export const run = async(
    sessionId: number, inputIndices: number[], inputTensors: TensorMetadata[], outputIndices: number[],
    outputTensors: Array<TensorMetadata|null>, options: InferenceSession.RunOptions): Promise<TensorMetadata[]> => {
  const wasm = getInstance();
  const session = activeSessions.get(sessionId);
  if (!session) {
    throw new Error(`cannot run inference. invalid session id: ${sessionId}`);
  }
  const [sessionHandle, inputNamesUTF8Encoded, outputNamesUTF8Encoded, ioBindingState] = session;

  const inputCount = inputIndices.length;
  const outputCount = outputIndices.length;

  let runOptionsHandle = 0;
  let runOptionsAllocs: number[] = [];

  const inputTensorHandles: number[] = [];
  const outputTensorHandles: number[] = [];
  const inputOutputAllocs: number[] = [];

  const beforeRunStack = wasm.stackSave();
  const inputValuesOffset = wasm.stackAlloc(inputCount * 4);
  const inputNamesOffset = wasm.stackAlloc(inputCount * 4);
  const outputValuesOffset = wasm.stackAlloc(outputCount * 4);
  const outputNamesOffset = wasm.stackAlloc(outputCount * 4);

  try {
    [runOptionsHandle, runOptionsAllocs] = setRunOptions(options);

    // create input tensors
    for (let i = 0; i < inputCount; i++) {
      prepareInputOutputTensor(inputTensors[i], inputTensorHandles, inputOutputAllocs, sessionId, inputIndices[i]);
    }

    // create output tensors
    for (let i = 0; i < outputCount; i++) {
      prepareInputOutputTensor(
          outputTensors[i], outputTensorHandles, inputOutputAllocs, sessionId, inputCount + outputIndices[i]);
    }

    let inputValuesIndex = inputValuesOffset / 4;
    let inputNamesIndex = inputNamesOffset / 4;
    let outputValuesIndex = outputValuesOffset / 4;
    let outputNamesIndex = outputNamesOffset / 4;
    for (let i = 0; i < inputCount; i++) {
      wasm.HEAPU32[inputValuesIndex++] = inputTensorHandles[i];
      wasm.HEAPU32[inputNamesIndex++] = inputNamesUTF8Encoded[inputIndices[i]];
    }
    for (let i = 0; i < outputCount; i++) {
      wasm.HEAPU32[outputValuesIndex++] = outputTensorHandles[i];
      wasm.HEAPU32[outputNamesIndex++] = outputNamesUTF8Encoded[outputIndices[i]];
    }

    if (!BUILD_DEFS.DISABLE_WEBGPU && ioBindingState) {
      const {handle, outputPreferredLocations, outputPreferredLocationsEncoded} = ioBindingState;

      if (inputNamesUTF8Encoded.length !== inputCount) {
        throw new Error(`input count from feeds (${
            inputCount}) is expected to be always equal to model's input count (${inputNamesUTF8Encoded.length}).`);
      }

      // process inputs
      for (let i = 0; i < inputCount; i++) {
        const index = inputIndices[i];
        const errorCode = await wasm._OrtBindInput(handle, inputNamesUTF8Encoded[index], inputTensorHandles[i]);
        if (errorCode !== 0) {
          checkLastError(`Can't bind input[${i}] for session=${sessionId}.`);
        }
      }

      // process pre-allocated outputs
      for (let i = 0; i < outputCount; i++) {
        const index = outputIndices[i];
        const location = outputTensors[i]?.[3];  // undefined means output is not pre-allocated.

        if (location) {
          // output is pre-allocated. bind the tensor.
          const errorCode = wasm._OrtBindOutput(handle, outputNamesUTF8Encoded[index], outputTensorHandles[i], 0);
          if (errorCode !== 0) {
            checkLastError(`Can't bind pre-allocated output[${i}] for session=${sessionId}.`);
          }
        } else {
          // output is not pre-allocated. reset preferred location.
          const errorCode =
              wasm._OrtBindOutput(handle, outputNamesUTF8Encoded[index], 0, outputPreferredLocationsEncoded[index]);
          if (errorCode !== 0) {
            checkLastError(`Can't bind output[${i}] to ${outputPreferredLocations[i]} for session=${sessionId}.`);
          }
        }
      }
    }

    let errorCode: number;

    if (!BUILD_DEFS.DISABLE_WEBGPU && ioBindingState) {
      errorCode = await wasm._OrtRunWithBinding(
          sessionHandle, ioBindingState.handle, outputCount, outputValuesOffset, runOptionsHandle);
    } else {
      errorCode = await wasm._OrtRun(
          sessionHandle, inputNamesOffset, inputValuesOffset, inputCount, outputNamesOffset, outputCount,
          outputValuesOffset, runOptionsHandle);
    }

    if (errorCode !== 0) {
      checkLastError('failed to call OrtRun().');
    }

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

        const preferredLocation = ioBindingState?.outputPreferredLocations[outputIndices[i]];

        if (type === 'string') {
          if (preferredLocation === 'gpu-buffer') {
            throw new Error('String tensor is not supported on GPU.');
          }
          const stringData: string[] = [];
          let dataIndex = dataOffset / 4;
          for (let i = 0; i < size; i++) {
            const offset = wasm.HEAPU32[dataIndex++];
            const maxBytesToRead = i === size - 1 ? undefined : wasm.HEAPU32[dataIndex] - offset;
            stringData.push(wasm.UTF8ToString(offset, maxBytesToRead));
          }
          output.push([type, dims, stringData, 'cpu']);
        } else {
          // If a certain output's preferred location is GPU but the tensor is empty, we still need to create a CPU
          // tensor for it. There is no mapping GPU buffer for an empty tensor.
          if (preferredLocation === 'gpu-buffer' && size > 0) {
            const gpuBuffer = wasm.jsepGetBuffer(dataOffset);
            const elementSize = getTensorElementSize(dataType);
            if (elementSize === undefined || !isGpuBufferSupportedType(type)) {
              throw new Error(`Unsupported data type: ${type}`);
            }

            // do not release the tensor right now. it will be released when user calls tensor.dispose().
            keepOutputTensor = true;

            output.push([
              type, dims, {
                gpuBuffer,
                download: wasm.jsepCreateDownloader(gpuBuffer, size * elementSize, type),
                dispose: () => {
                  wasm._OrtReleaseTensor(tensor);
                }
              },
              'gpu-buffer'
            ]);
          } else {
            const typedArrayConstructor = tensorTypeToTypedArrayConstructor(type);
            const data = new typedArrayConstructor(size);
            new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
                .set(wasm.HEAPU8.subarray(dataOffset, dataOffset + data.byteLength));
            output.push([type, dims, data, 'cpu']);
          }
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

    if (ioBindingState) {
      wasm._OrtClearBoundOutputs(ioBindingState.handle);
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
    checkLastError('Can\'t get an profile file name.');
  }
  wasm._OrtFree(profileFileName);
};

export const extractTransferableBuffers = (tensors: readonly SerializableTensorMetadata[]): ArrayBufferLike[] => {
  const buffers: ArrayBufferLike[] = [];
  for (const tensor of tensors) {
    const data = tensor[2];
    if (!Array.isArray(data) && 'buffer' in data) {
      buffers.push(data.buffer);
    }
  }
  return buffers;
};
