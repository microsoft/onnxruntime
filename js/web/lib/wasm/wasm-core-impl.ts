// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession, Tensor} from 'onnxruntime-common';

import {SerializableModeldata, SerializableSessionMetadata, SerializableTensor} from './proxy-messages';
import {setRunOptions} from './run-options';
import {setSessionOptions} from './session-options';
import {allocWasmString} from './string-utils';
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
        wasm._OrtReleaseSessionOptions(sessionOptionsHandle);
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
 * Copied from ONNX definition. Use this to drop dependency 'onnx_proto' to decrease compiled .js file size.
 */
const enum DataType {
  undefined = 0,
  float = 1,
  uint8 = 2,
  int8 = 3,
  uint16 = 4,
  int16 = 5,
  int32 = 6,
  int64 = 7,
  string = 8,
  bool = 9,
  float16 = 10,
  double = 11,
  uint32 = 12,
  uint64 = 13,
  complex64 = 14,
  complex128 = 15,
  bfloat16 = 16
}


const tensorDataTypeStringToEnum = (type: string): DataType => {
  switch (type) {
    case 'int8':
      return DataType.int8;
    case 'uint8':
      return DataType.uint8;
    case 'bool':
      return DataType.bool;
    case 'int16':
      return DataType.int16;
    case 'uint16':
      return DataType.uint16;
    case 'int32':
      return DataType.int32;
    case 'uint32':
      return DataType.uint32;
    case 'float32':
      return DataType.float;
    case 'float64':
      return DataType.double;
    case 'string':
      return DataType.string;
    case 'int64':
      return DataType.int64;
    case 'uint64':
      return DataType.uint64;

    default:
      throw new Error(`unsupported data type: ${type}`);
  }
};

const tensorDataTypeEnumToString = (typeProto: DataType): Tensor.Type => {
  switch (typeProto) {
    case DataType.int8:
      return 'int8';
    case DataType.uint8:
      return 'uint8';
    case DataType.bool:
      return 'bool';
    case DataType.int16:
      return 'int16';
    case DataType.uint16:
      return 'uint16';
    case DataType.int32:
      return 'int32';
    case DataType.uint32:
      return 'uint32';
    case DataType.float:
      return 'float32';
    case DataType.double:
      return 'float64';
    case DataType.string:
      return 'string';
    case DataType.int64:
      return 'int64';
    case DataType.uint64:
      return 'uint64';

    default:
      throw new Error(`unsupported data type: ${typeProto}`);
  }
};

const numericTensorTypeToTypedArray = (type: Tensor.Type): Float32ArrayConstructor|Uint8ArrayConstructor|
    Int8ArrayConstructor|Uint16ArrayConstructor|Int16ArrayConstructor|Int32ArrayConstructor|BigInt64ArrayConstructor|
    Uint8ArrayConstructor|Float64ArrayConstructor|Uint32ArrayConstructor|BigUint64ArrayConstructor => {
      switch (type) {
        case 'float32':
          return Float32Array;
        case 'uint8':
          return Uint8Array;
        case 'int8':
          return Int8Array;
        case 'uint16':
          return Uint16Array;
        case 'int16':
          return Int16Array;
        case 'int32':
          return Int32Array;
        case 'bool':
          return Uint8Array;
        case 'float64':
          return Float64Array;
        case 'uint32':
          return Uint32Array;
        case 'int64':
          return BigInt64Array;
        case 'uint64':
          return BigUint64Array;
        default:
          throw new Error(`unsupported type: ${type}`);
      }
    };

/**
 * perform inference run
 */
export const run =
    (sessionId: number, inputIndices: number[], inputs: SerializableTensor[], outputIndices: number[],
     options: InferenceSession.RunOptions): SerializableTensor[] => {
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
                  const typedArrayConstructor = numericTensorTypeToTypedArray(type);
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
