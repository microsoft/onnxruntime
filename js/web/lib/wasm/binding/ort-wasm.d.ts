// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

declare namespace JSEP {
  type BackendType = unknown;
  type AllocFunction = (size: number) => number;
  type FreeFunction = (size: number) => number;
  type UploadFunction = (dataOffset: number, gpuDataId: number, size: number) => void;
  type DownloadFunction = (gpuDataId: number, dataOffset: number, size: number) => Promise<void>;
  type CreateKernelFunction = (name: string, kernel: number, attribute: unknown) => void;
  type ReleaseKernelFunction = (kernel: number) => void;
  type RunFunction = (kernel: number, contextDataOffset: number) => number;
}

export interface OrtWasmModule extends EmscriptenModule {
  // #region emscripten functions
  stackSave(): number;
  stackRestore(stack: number): void;
  stackAlloc(size: number): number;

  UTF8ToString(offset: number, maxBytesToRead?: number): string;
  lengthBytesUTF8(str: string): number;
  stringToUTF8(str: string, offset: number, maxBytes: number): void;
  // #endregion

  // #region ORT APIs
  _OrtInit(numThreads: number, loggingLevel: number): number;

  _OrtGetLastError(errorCodeOffset: number, errorMessageOffset: number): void;

  _OrtCreateSession(dataOffset: number, dataLength: number, sessionOptionsHandle: number): number;
  _OrtReleaseSession(sessionHandle: bigint): void;
  _OrtGetInputOutputCount(sessionHandle: bigint, inputCountOffset: number, outputCountOffset: number): number;
  _OrtGetInputName(sessionHandle: bigint, index: number): number;
  _OrtGetOutputName(sessionHandle: bigint, index: number): number;

  _OrtFree(stringHandle: bigint): void;

  _OrtCreateTensor(dataType: number, dataOffset: bigint, dataLength: bigint, dimsOffset: bigint, dimsLength: bigint):
      bigint;
  _OrtGetTensorData(tensorHandle: bigint, dataType: bigint, dataOffset: bigint, dimsOffset: bigint, dimsLength: bigint):
      number;
  _OrtReleaseTensor(tensorHandle: bigint): void;
  _OrtRun(
      sessionHandle: bigint, inputNamesOffset: bigint, inputsOffset: bigint, inputCount: bigint,
      outputNamesOffset: bigint, outputCount: bigint, outputsOffset: bigint, runOptionsHandle: bigint): number;

  _OrtCreateSessionOptions(
      graphOptimizationLevel: number, enableCpuMemArena: boolean, enableMemPattern: boolean, executionMode: number,
      enableProfiling: boolean, profileFilePrefix: number, logId: number, logSeverityLevel: number,
      logVerbosityLevel: number, optimizedModelFilePath: number): bigint;
  _OrtAppendExecutionProvider(sessionOptionsHandle: bigint, name: number): bigint;
  _OrtAddSessionConfigEntry(sessionOptionsHandle: bigint, configKey: bigint, configValue: bigint): bigint;
  _OrtReleaseSessionOptions(sessionOptionsHandle: bigint): void;

  _OrtCreateRunOptions(logSeverityLevel: number, logVerbosityLevel: number, terminate: boolean, tag: number): bigint;
  _OrtAddRunConfigEntry(runOptionsHandle: bigint, configKey: bigint, configValue: bigint): number;
  _OrtReleaseRunOptions(runOptionsHandle: bigint): void;

  _OrtEndProfiling(sessionHandle: bigint): bigint;
  // #endregion

  // #region config
  mainScriptUrlOrBlob?: string|Blob;
  // #endregion

  // #region JSEP
  jsepInit?
      (backend: JSEP.BackendType, alloc: JSEP.AllocFunction, free: JSEP.FreeFunction, upload: JSEP.UploadFunction,
       download: JSEP.DownloadFunction, createKernel: JSEP.CreateKernelFunction,
       releaseKernel: JSEP.ReleaseKernelFunction, run: JSEP.RunFunction): void;

  _JsepOutput(context: bigint, index: bigint, data: bigint): bigint;

  jsepRunPromise?: Promise<number>;
  // #endregion
}

declare const moduleFactory: EmscriptenModuleFactory<OrtWasmModule>;
export default moduleFactory;
