// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {onnx} from 'onnx-proto';
import {env, InferenceSession, SessionHandler, Tensor, TypedTensor} from 'onnxruntime-common';

import {setRunOptions} from './run-options';
import {setSessionOptions} from './session-options';
import {allocWasmString} from './string-utils';
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

const getLogLevel = (logLevel: 'verbose'|'info'|'warning'|'error'|'fatal'): number => {
  switch (logLevel) {
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
      throw new Error(`unsupported logging level: ${logLevel}`);
  }
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
      const errorCode = wasm._OrtInit(env.wasm.numThreads!, getLogLevel(env.logLevel!));
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
      // if (tensor.type === 'string') {
      //   // TODO: support string tensor
      //   throw new TypeError('string tensor is not supported');
      // }
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
    let runOptionsAllocs: number[] = [];

    const inputValues: number[] = [];
    const inputAllocs: number[] = [];

    try {
      [runOptionsHandle, runOptionsAllocs] = setRunOptions(options);

      // create input tensors
      for (let i = 0; i < inputCount; i++) {
        const data = inputArray[i].data;

        let dataOffset: number;
        let dataByteLength: number;

        if (Array.isArray(data)) {
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

        const dims = inputArray[i].dims;

        const stack = wasm.stackSave();
        const dimsOffset = wasm.stackAlloc(4 * dims.length);
        try {
          let dimIndex = dimsOffset / 4;
          dims.forEach(d => wasm.HEAP32[dimIndex++] = d);
          const tensor = wasm._OrtCreateTensor(
              tensorDataTypeStringToEnum(inputArray[i].type), dataOffset, dataByteLength, dimsOffset, dims.length);
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

            let type: Tensor.Type|undefined, dataOffset = 0;
            try {
              errorCode = wasm._OrtGetTensorData(
                  tensor, tensorDataOffset, tensorDataOffset + 4, tensorDataOffset + 8, tensorDataOffset + 12);
              if (errorCode !== 0) {
                throw new Error(`Can't get a tensor data. error code = ${errorCode}`);
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
                output[this.outputNames[outputIndices[i]]] = new Tensor('string', stringData, dims);
              } else {
                const typedArray = numericTensorTypeToTypedArray(type);
                const t = new Tensor(type, new typedArray(size), dims) as TypedTensor<Exclude<Tensor.Type, 'string'>>;
                new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength)
                    .set(wasm.HEAPU8.subarray(dataOffset, dataOffset + t.data.byteLength));
                output[this.outputNames[outputIndices[i]]] = t;
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
  }

  startProfiling(): void {
    // TODO: implement profiling
  }

  endProfiling(): void {
    // TODO: implement profiling
  }
}
