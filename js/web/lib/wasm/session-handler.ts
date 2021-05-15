// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {onnx} from 'onnx-proto';
import {env, InferenceSession, SessionHandler, Tensor, TypedTensor} from 'onnxruntime-common';
import {getInstance} from './wasm-factory';

let ortInit: boolean;

const tensorDataTypeStringToEnum = (type: string): onnx.TensorProto.DataType => {
  switch (type) {
    case 'int8':
      return onnx.TensorProto.DataType.INT8;
    case 'uint8':
      return onnx.TensorProto.DataType.UINT8;
    case 'bool':
      return onnx.TensorProto.DataType.BOOL;
    case 'int16':
      return onnx.TensorProto.DataType.INT16;
    case 'uint16':
      return onnx.TensorProto.DataType.UINT16;
    case 'int32':
      return onnx.TensorProto.DataType.INT32;
    case 'uint32':
      return onnx.TensorProto.DataType.UINT32;
    case 'float32':
      return onnx.TensorProto.DataType.FLOAT;
    case 'float64':
      return onnx.TensorProto.DataType.DOUBLE;
    case 'string':
      return onnx.TensorProto.DataType.STRING;
    case 'int64':
      return onnx.TensorProto.DataType.INT64;
    case 'uint64':
      return onnx.TensorProto.DataType.UINT64;

    default:
      throw new Error(`unsupported data type: ${type}`);
  }
};

const tensorDataTypeEnumToString = (typeProto: onnx.TensorProto.DataType): Tensor.Type => {
  switch (typeProto) {
    case onnx.TensorProto.DataType.INT8:
      return 'int8';
    case onnx.TensorProto.DataType.UINT8:
      return 'uint8';
    case onnx.TensorProto.DataType.BOOL:
      return 'bool';
    case onnx.TensorProto.DataType.INT16:
      return 'int16';
    case onnx.TensorProto.DataType.UINT16:
      return 'uint16';
    case onnx.TensorProto.DataType.INT32:
      return 'int32';
    case onnx.TensorProto.DataType.UINT32:
      return 'uint32';
    case onnx.TensorProto.DataType.FLOAT:
      return 'float32';
    case onnx.TensorProto.DataType.DOUBLE:
      return 'float64';
    case onnx.TensorProto.DataType.STRING:
      return 'string';
    case onnx.TensorProto.DataType.INT64:
      return 'int32';
    case onnx.TensorProto.DataType.UINT64:
      return 'uint32';

    default:
      throw new Error(`unsupported data type: ${onnx.TensorProto.DataType[typeProto]}`);
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

const getLoggingLevel = (loggingLevel: 'verbose'|'info'|'warning'|'error'|'fatal'): number => {
  switch (loggingLevel) {
    case 'verbose':
      return 0;
    case 'info':
      return 1;
    case 'warning':
      return 2;
    case 'error':
      return 3;
    case 'fatal':
      return 4;
    default:
      throw new Error(`unsupported logging level: ${loggingLevel}`);
  }
};

const setSessionOptions = (options?: InferenceSession.SessionOptions): [number, number[]] => {
  const wasm = getInstance();
  const sessionOptionsHandle = wasm._OrtCreateSessionOptions();
  const allocs: number[] = [];

  if (sessionOptionsHandle === 0) {
    throw new Error('Can\'t create session options');
  }

  if (options === undefined) {
    return [sessionOptionsHandle, allocs];
  }

  let errorCode = 0;

  if (options.graphOptimizationLevel !== undefined) {
    switch (options.graphOptimizationLevel) {
      case 'disabled':
        errorCode = wasm._OrtSetSessionGraphOptimizationLevel(sessionOptionsHandle, 0);
        break;
      case 'basic':
        errorCode = wasm._OrtSetSessionGraphOptimizationLevel(sessionOptionsHandle, 1);
        break;
      case 'extended':
        errorCode = wasm._OrtSetSessionGraphOptimizationLevel(sessionOptionsHandle, 2);
        break;
      case 'all':
        errorCode = wasm._OrtSetSessionGraphOptimizationLevel(sessionOptionsHandle, 99);
        break;
      default:
        throw new Error(`unsupported graph optimization level: ${options.graphOptimizationLevel}`);
    }
    if (errorCode !== 0) {
      throw new Error(`Can't set a graph optimization level as a session option. error code = ${errorCode}`);
    }
  }

  if (options.enableCpuMemArena !== undefined) {
    if (options.enableCpuMemArena) {
      errorCode = wasm._OrtEnableCpuMemArena(sessionOptionsHandle);
    } else {
      errorCode = wasm._OrtDisableCpuMemArena(sessionOptionsHandle);
    }
    if (errorCode !== 0) {
      throw new Error(`Can't set a CPU memory arena as a session option. error code = ${errorCode}`);
    }
  }

  if (options.enableMemPattern !== undefined) {
    if (options.enableMemPattern) {
      errorCode = wasm._OrtEnableMemPattern(sessionOptionsHandle);
    } else {
      errorCode = wasm._OrtDisableMemPattern(sessionOptionsHandle);
    }
    if (errorCode !== 0) {
      throw new Error(`Can't set a memory pattern as a session option. error code = ${errorCode}`);
    }
  }

  if (options.executionMode !== undefined) {
    switch (options.executionMode) {
      case 'sequential':
        errorCode = wasm._OrtSetSessionExecutionMode(sessionOptionsHandle, 0);
        break;
      case 'parallel':
        errorCode = wasm._OrtSetSessionExecutionMode(sessionOptionsHandle, 1);
        break;
      default:
        throw new Error(`unsupported execution mode: ${options.executionMode}`);
    }
    if (errorCode !== 0) {
      throw new Error(`Can't set an execution mode as a session option. error code = ${errorCode}`);
    }
  }

  if (options.logId !== undefined) {
    const logIdDataLength = wasm.lengthBytesUTF8(options.logId) + 1;
    const logIdDataOffset = wasm._malloc(logIdDataLength);
    wasm.stringToUTF8(options.logId, logIdDataOffset, logIdDataLength);
    errorCode = wasm._OrtSetSessionLogId(sessionOptionsHandle, logIdDataOffset);
    allocs.push(logIdDataOffset);
    if (errorCode !== 0) {
      throw new Error(`Can't set a log id as a session option. error code = ${errorCode}`);
    }
  }

  if (options.logSeverityLevel !== undefined) {
    errorCode = wasm._OrtSetSessionLogSeverityLevel(sessionOptionsHandle, options.logSeverityLevel);
    if (errorCode !== 0) {
      throw new Error(`Can't set a log severity level as a session option. error code = ${errorCode}`);
    }
  }

  return [sessionOptionsHandle, allocs];
};

const setRunOptions = (options: InferenceSession.RunOptions): [number, number[]] => {
  const wasm = getInstance();
  const runOptionsHandle = wasm._OrtCreateRunOptions();
  if (runOptionsHandle === 0) {
    throw new Error('Can\'t create run options');
  }

  const allocs: number[] = [];
  let errorCode = 0;

  if (options.logSeverityLevel !== undefined) {
    errorCode = wasm._OrtRunOptionsSetRunLogSeverityLevel(runOptionsHandle, options.logSeverityLevel);
    if (errorCode !== 0) {
      throw new Error(`Can't set a log severity level as a run option. error code = ${errorCode}`);
    }
  }

  if (options.tag !== undefined) {
    const tagDataLength = wasm.lengthBytesUTF8(options.tag) + 1;
    const tagDataOffset = wasm._malloc(tagDataLength);
    wasm.stringToUTF8(options.tag, tagDataOffset, tagDataLength);
    errorCode = wasm._OrtRunOptionsSetRunTag(runOptionsHandle, tagDataOffset);
    allocs.push(tagDataOffset);
    if (errorCode !== 0) {
      throw new Error(`Can't set a tag as a run option. error code = ${errorCode}`);
    }
  }

  return [runOptionsHandle, allocs];
};

export class OnnxruntimeWebAssemblySessionHandler implements SessionHandler {
  private sessionHandle: number;

  inputNames: string[];
  private inputNamesUTF8Encoded: number[];
  outputNames: string[];
  private outputNamesUTF8Encoded: number[];

  loadModel(model: Uint8Array, options?: InferenceSession.SessionOptions): void {
    const wasm = getInstance();
    if (!ortInit) {
      const errorCode = wasm._OrtInit(env.wasm.numThreads!, getLoggingLevel(env.wasm.loggingLevel!));
      if (errorCode !== 0) {
        throw new Error(`Can't initialize onnxruntime. error code = ${errorCode}`);
      }
      ortInit = true;
    }

    const modelDataOffset = wasm._malloc(model.byteLength);
    let sessionOptionsHandle = 0;
    let allocs: number[] = [];

    try {
      [sessionOptionsHandle, allocs] = setSessionOptions(options);

      wasm.HEAPU8.set(model, modelDataOffset);
      this.sessionHandle = wasm._OrtCreateSession(modelDataOffset, model.byteLength, sessionOptionsHandle);
      if (this.sessionHandle === 0) {
        throw new Error('Can\'t create a session');
      }
    } finally {
      wasm._free(modelDataOffset);
      wasm._OrtReleaseSessionOptions(sessionOptionsHandle);
      allocs.forEach(wasm._free);
    }

    const inputCount = wasm._OrtGetInputCount(this.sessionHandle);
    const outputCount = wasm._OrtGetOutputCount(this.sessionHandle);

    this.inputNames = [];
    this.inputNamesUTF8Encoded = [];
    this.outputNames = [];
    this.outputNamesUTF8Encoded = [];
    for (let i = 0; i < inputCount; i++) {
      const name = wasm._OrtGetInputName(this.sessionHandle, i);
      if (name === 0) {
        throw new Error('Can\'t get an input name');
      }
      this.inputNamesUTF8Encoded.push(name);
      this.inputNames.push(wasm.UTF8ToString(name));
    }
    for (let i = 0; i < outputCount; i++) {
      const name = wasm._OrtGetOutputName(this.sessionHandle, i);
      if (name === 0) {
        throw new Error('Can\'t get an output name');
      }
      this.outputNamesUTF8Encoded.push(name);
      this.outputNames.push(wasm.UTF8ToString(name));
    }
  }

  async dispose(): Promise<void> {
    const wasm = getInstance();
    if (this.inputNamesUTF8Encoded) {
      this.inputNamesUTF8Encoded.forEach(wasm._OrtFree);
      this.inputNamesUTF8Encoded = [];
    }
    if (this.outputNamesUTF8Encoded) {
      this.outputNamesUTF8Encoded.forEach(wasm._OrtFree);
      this.outputNamesUTF8Encoded = [];
    }
    if (this.sessionHandle) {
      wasm._OrtReleaseSession(this.sessionHandle);
      this.sessionHandle = 0;
    }
  }

  async run(feeds: SessionHandler.FeedsType, fetches: SessionHandler.FetchesType, options: InferenceSession.RunOptions):
      Promise<SessionHandler.ReturnType> {
    const wasm = getInstance();

    const inputArray: Tensor[] = [];
    const inputIndices: number[] = [];
    Object.entries(feeds).forEach(kvp => {
      const name = kvp[0];
      const tensor = kvp[1];
      const index = this.inputNames.indexOf(name);
      if (index === -1) {
        throw new Error(`invalid input '${name}'`);
      }
      if (tensor.type === 'string') {
        // TODO: support string tensor
        throw new TypeError('string tensor is not supported');
      }
      inputArray.push(tensor);
      inputIndices.push(index);
    });

    const outputIndices: number[] = [];
    Object.entries(fetches).forEach(kvp => {
      const name = kvp[0];
      // TODO: support pre-allocated output
      const index = this.outputNames.indexOf(name);
      if (index === -1) {
        throw new Error(`invalid output '${name}'`);
      }
      outputIndices.push(index);
    });

    const inputCount = inputIndices.length;
    const outputCount = outputIndices.length;

    let runOptionsHandle = 0;
    let allocs: number[] = [];

    const inputValues: number[] = [];
    const inputDataOffsets: number[] = [];

    try {
      [runOptionsHandle, allocs] = setRunOptions(options);

      // create input tensors
      for (let i = 0; i < inputCount; i++) {
        const data = inputArray[i].data;
        if (Array.isArray(data)) {
          // string tensor
          throw new TypeError('string tensor is not supported');
        } else {
          const dataOffset = wasm._malloc(data.byteLength);
          inputDataOffsets.push(dataOffset);
          wasm.HEAPU8.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), dataOffset);

          const dims = inputArray[i].dims;

          const stack = wasm.stackSave();
          const dimsOffset = wasm.stackAlloc(4 * dims.length);
          try {
            let dimIndex = dimsOffset / 4;
            dims.forEach(d => wasm.HEAP32[dimIndex++] = d);
            const tensor = wasm._OrtCreateTensor(
                tensorDataTypeStringToEnum(inputArray[i].type), dataOffset, data.byteLength, dimsOffset, dims.length);
            if (tensor === 0) {
              throw new Error('Can\'t create a tensor');
            }
            inputValues.push(tensor);
          } finally {
            wasm.stackRestore(stack);
          }
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
          wasm.HEAPU32[inputNamesIndex++] = this.inputNamesUTF8Encoded[inputIndices[i]];
        }
        for (let i = 0; i < outputCount; i++) {
          wasm.HEAPU32[outputValuesIndex++] = 0;
          wasm.HEAPU32[outputNamesIndex++] = this.outputNamesUTF8Encoded[outputIndices[i]];
        }

        // support RunOptions
        let errorCode = wasm._OrtRun(
            this.sessionHandle, inputNamesOffset, inputValuesOffset, inputCount, outputNamesOffset, outputCount,
            outputValuesOffset, runOptionsHandle);

        const output: {[name: string]: Tensor} = {};

        if (errorCode === 0) {
          for (let i = 0; i < outputCount; i++) {
            const tensor = wasm.HEAPU32[outputValuesOffset / 4 + i];

            const beforeGetTensorDataStack = wasm.stackSave();
            // stack allocate 4 pointer value
            const tensorDataOffset = wasm.stackAlloc(4 * 4);
            try {
              errorCode = wasm._OrtGetTensorData(
                  tensor, tensorDataOffset, tensorDataOffset + 4, tensorDataOffset + 8, tensorDataOffset + 12);
              if (errorCode !== 0) {
                throw new Error(`Can't get a tensor data. error code = ${errorCode}`);
              }
              let tensorDataIndex = tensorDataOffset / 4;
              const dataType = wasm.HEAPU32[tensorDataIndex++];
              const dataOffset: number = wasm.HEAPU32[tensorDataIndex++];
              const dimsOffset = wasm.HEAPU32[tensorDataIndex++];
              const dimsLength = wasm.HEAPU32[tensorDataIndex++];
              const dims = [];
              for (let i = 0; i < dimsLength; i++) {
                dims.push(wasm.HEAPU32[dimsOffset / 4 + i]);
              }
              wasm._OrtFree(dimsOffset);

              const type = tensorDataTypeEnumToString(dataType);
              if (type === 'string') {
                // string tensor
                throw new TypeError('string tensor is not supported');
              } else {
                const typedArray = numericTensorTypeToTypedArray(type);
                const size = dims.length === 0 ? 1 : dims.reduce((a, b) => a * b);
                const t = new Tensor(type, new typedArray(size), dims) as TypedTensor<Exclude<Tensor.Type, 'string'>>;
                new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength)
                    .set(wasm.HEAPU8.subarray(dataOffset, dataOffset + t.data.byteLength));
                output[this.outputNames[outputIndices[i]]] = t;
              }
            } finally {
              wasm.stackRestore(beforeGetTensorDataStack);
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
      inputDataOffsets.forEach(wasm._free);

      wasm._OrtReleaseRunOptions(runOptionsHandle);
      allocs.forEach(wasm._free);
    }
  }

  startProfiling(): void {
    // TODO: implement profiling
  }

  endProfiling(): void {
    // TODO: implement profiling
  }
}
