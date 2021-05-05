// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

export interface BackendWasmModule extends EmscriptenModule {
  stackSave(): number;
  stackRestore(stack: number): void;
  stackAlloc(size: number): number;

  UTF8ToString(offset: number): string;
  lengthBytesUTF8(str: string): number;
  stringToUTF8(str: string, offset: number, maxBytes: number): void;

  _OrtInit(): void;

  _OrtCreateSession(dataOffset: number, dataLength: number): number;
  _OrtReleaseSession(sessionHandle: number): void;
  _OrtGetInputCount(sessionHandle: number): number;
  _OrtGetOutputCount(sessionHandle: number): number;
  _OrtGetInputName(sessionHandle: number, index: number): number;
  _OrtGetOutputName(sessionHandle: number, index: number): number;

  _OrtFree(stringHandle: number): void;

  _OrtCreateTensor(dataType: number, dataOffset: number, dataLength: number, dimsOffset: number, dimsLength: number):
      number;
  _OrtGetTensorData(tensorHandle: number, dataType: number, dataOffset: number, dimsOffset: number, dimsLength: number):
      void;
  _OrtReleaseTensor(tensorHandle: number): void;
  _OrtRun(
      sessionHandle: number, inputNamesOffset: number, inputsOffset: number, inputCount: number,
      outputNamesOffset: number, outputCount: number, outputsOffset: number): number;
}

declare const moduleFactory: EmscriptenModuleFactory<BackendWasmModule>;
export default moduleFactory;
