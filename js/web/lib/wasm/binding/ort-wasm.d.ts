// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

export interface OrtWasmModule extends EmscriptenModule {
  //#region emscripten functions
  stackSave(): number;
  stackRestore(stack: number): void;
  stackAlloc(size: number): number;

  UTF8ToString(offset: number): string;
  lengthBytesUTF8(str: string): number;
  stringToUTF8(str: string, offset: number, maxBytes: number): void;
  //#endregion

  //#region ORT APIs
  _OrtInit(numThreads: number, loggingLevel: number): void;

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
  //#endregion

  //#region config
  mainScriptUrlOrBlob?: string|Blob;
  //#endregion
}

declare const moduleFactory: EmscriptenModuleFactory<OrtWasmModule>;
export default moduleFactory;
