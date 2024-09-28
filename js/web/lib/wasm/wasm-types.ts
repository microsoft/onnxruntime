// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// WebNN API currently does not have a TypeScript definition file. This file is a workaround with types generated from
// WebNN API specification.
// https://github.com/webmachinelearning/webnn/issues/677
/// <reference path="jsep/webnn/webnn.d.ts" />

import type { Tensor } from 'onnxruntime-common';
import { DataType } from './wasm-common';

/* eslint-disable @typescript-eslint/naming-convention */

export declare namespace JSEP {
  type BackendType = unknown;
  type AllocFunction = (size: number) => number;
  type FreeFunction = (size: number) => number;
  type UploadFunction = (dataOffset: number, gpuDataId: number, size: number) => void;
  type DownloadFunction = (gpuDataId: number, dataOffset: number, size: number) => Promise<void>;
  type CreateKernelFunction = (name: string, kernel: number, attribute: unknown) => void;
  type ReleaseKernelFunction = (kernel: number) => void;
  type RunFunction = (
    kernel: number,
    contextDataOffset: number,
    sessionHandle: number,
    errors: Array<Promise<string | null>>,
  ) => number;
  type CaptureBeginFunction = () => void;
  type CaptureEndFunction = () => void;
  type ReplayFunction = () => void;
  type ReserveTensorIdFunction = () => number;
  type ReleaseTensorIdFunction = (tensorId: number) => void;
  type EnsureTensorFunction = (
    tensorId: number,
    dataType: DataType,
    shape: readonly number[],
    copyOld: boolean,
  ) => Promise<MLTensor>;
  type UploadTensorFunction = (tensorId: number, data: Uint8Array) => void;
  type DownloadTensorFunction = (tensorId: number, dstBuffer: ArrayBufferView | ArrayBuffer) => Promise<undefined>;

  export interface Module extends WebGpuModule, WebNnModule {
    /**
     * Mount the external data file to an internal map, which will be used during session initialization.
     *
     * @param externalDataFilePath - specify the relative path of the external data file.
     * @param externalDataFileData - specify the content data.
     */
    mountExternalData(externalDataFilePath: string, externalDataFileData: Uint8Array): void;
    /**
     * Unmount all external data files from the internal map.
     */
    unmountExternalData(): void;

    /**
     * This is the entry of JSEP initialization. This function is called once when initializing ONNX Runtime per
     * backend. This function initializes Asyncify support. If name is 'webgpu', also initializes WebGPU backend and
     * registers a few callbacks that will be called in C++ code.
     */
    jsepInit(
      name: 'webgpu',
      initParams: [
        backend: BackendType,
        alloc: AllocFunction,
        free: FreeFunction,
        upload: UploadFunction,
        download: DownloadFunction,
        createKernel: CreateKernelFunction,
        releaseKernel: ReleaseKernelFunction,
        run: RunFunction,
        captureBegin: CaptureBeginFunction,
        captureEnd: CaptureEndFunction,
        replay: ReplayFunction,
      ],
    ): void;
    jsepInit(
      name: 'webnn',
      initParams: [
        backend: BackendType,
        reserveTensorId: ReserveTensorIdFunction,
        releaseTensorId: ReleaseTensorIdFunction,
        ensureTensor: EnsureTensorFunction,
        uploadTensor: UploadTensorFunction,
        downloadTensor: DownloadTensorFunction,
      ],
    ): void;
  }

  export interface WebGpuModule {
    /**
     * [exported from wasm] Specify a kernel's output when running OpKernel::Compute().
     *
     * @param context - specify the kernel context pointer.
     * @param index - specify the index of the output.
     * @param data - specify the pointer to encoded data of type and dims.
     */
    _JsepOutput(context: number, index: number, data: number): number;
    /**
     * [exported from wasm] Get name of an operator node.
     *
     * @param kernel - specify the kernel pointer.
     * @returns the pointer to a C-style UTF8 encoded string representing the node name.
     */
    _JsepGetNodeName(kernel: number): number;

    /**
     * [exported from pre-jsep.js] Register a user GPU buffer for usage of a session's input or output.
     *
     * @param sessionId - specify the session ID.
     * @param index - specify an integer to represent which input/output it is registering for. For input, it is the
     *     input_index corresponding to the session's inputNames. For output, it is the inputCount + output_index
     *     corresponding to the session's ouputNames.
     * @param buffer - specify the GPU buffer to register.
     * @param size - specify the original data size in byte.
     * @returns the GPU data ID for the registered GPU buffer.
     */
    jsepRegisterBuffer: (sessionId: number, index: number, buffer: GPUBuffer, size: number) => number;
    /**
     * [exported from pre-jsep.js] Get the GPU buffer by GPU data ID.
     *
     * @param dataId - specify the GPU data ID
     * @returns the GPU buffer.
     */
    jsepGetBuffer: (dataId: number) => GPUBuffer;
    /**
     * [exported from pre-jsep.js] Create a function to be used to create a GPU Tensor.
     *
     * @param gpuBuffer - specify the GPU buffer
     * @param size - specify the original data size in byte.
     * @param type - specify the tensor type.
     * @returns the generated downloader function.
     */
    jsepCreateDownloader: (
      gpuBuffer: GPUBuffer,
      size: number,
      type: Tensor.GpuBufferDataTypes,
    ) => () => Promise<Tensor.DataTypeMap[Tensor.GpuBufferDataTypes]>;
    /**
     *  [exported from pre-jsep.js] Called when InferenceSession.run started. This function will be called before
     * _OrtRun[WithBinding]() is called.
     * @param sessionId - specify the session ID.
     */
    jsepOnRunStart: (sessionId: number) => void;
    /**
     * [exported from pre-jsep.js] Release a session. This function will be called before _OrtReleaseSession() is
     * called.
     * @param sessionId - specify the session ID.
     * @returns
     */
    jsepOnReleaseSession: (sessionId: number) => void;
  }

  export interface WebNnModule {
    /**
     * Active MLContext used to create WebNN EP.
     */
    currentContext: MLContext;

    /**
     * Disables creating MLTensors. This is used to avoid creating MLTensors for graph initializers.
     */
    shouldTransferToMLTensor: boolean;

    /**
     * [exported from pre-jsep.js] Register MLContext for a session.
     * @param sessionId - specify the session ID.
     * @param context - specify the MLContext.
     * @returns
     */
    jsepRegisterMLContext: (sessionId: number, context: MLContext) => void;
    /**
     * [exported from pre-jsep.js] Reserve a MLTensor ID attached to the current session.
     * @returns the MLTensor ID.
     */
    jsepReserveTensorId: () => number;
    /**
     * [exported from pre-jsep.js] Release an MLTensor ID from use and destroys underlying MLTensor if no longer in use.
     * @param tensorId - specify the MLTensor ID.
     * @returns
     */
    jsepReleaseTensorId: (tensorId: number) => void;
    /**
     * [exported from pre-jsep.js] Ensure that an MLTensor of a given type and shape exists for a MLTensor ID.
     * @param tensorId - specify the MLTensor ID.
     * @param onnxDataType - specify the data type.
     * @param shape - specify the dimensions (WebNN shape) of the tensor.
     * @param copyOld - specify whether to copy the old tensor if a new tensor was created.
     * @returns the MLTensor associated with the tensor ID.
     */
    jsepEnsureTensor: (tensorId: number, dataType: DataType, shape: number[], copyOld: boolean) => Promise<MLTensor>;
    /**
     * [exported from pre-jsep.js] Upload data to an MLTensor.
     * @param tensorId - specify the MLTensor ID.
     * @param data - specify the data to upload. It can be a TensorProto::data_type or a WebNN MLOperandDataType.
     * @returns
     */
    jsepUploadTensor: (tensorId: number, data: Uint8Array) => void;
    /**
     * [exported from pre-jsep.js] Download data from an MLTensor.
     * @param tensorId - specify the MLTensor ID.
     * @returns the downloaded data.
     */
    jsepDownloadTensor: (tensorId: number, dstBuffer: ArrayBufferView | ArrayBuffer) => Promise<undefined>;
    /**
     * [exported from pre-jsep.js] Creates a downloader function to download data from an MLTensor.
     * @param tensorId - specify the MLTensor ID.
     * @param type - specify the data type.
     * @returns the downloader function.
     */
    jsepCreateMLTensorDownloader: (
      tensorId: number,
      type: Tensor.MLTensorDataTypes,
    ) => () => Promise<Tensor.DataTypeMap[Tensor.MLTensorDataTypes]>;
    /**
     * [exported from pre-jsep.js] Registers an external MLTensor to a session.
     * @param tensor - specify the MLTensor.
     * @param dataType - specify the data type.
     * @param dimensions - specify the dimensions.
     * @returns the MLTensor ID for the external MLTensor.
     */
    jsepRegisterMLTensor: (tensor: MLTensor, onnxDataType: DataType, dimensions: readonly number[]) => number;
  }
}

export interface OrtInferenceAPIs {
  _OrtInit(numThreads: number, loggingLevel: number): number;

  _OrtGetLastError(errorCodeOffset: number, errorMessageOffset: number): void;

  _OrtCreateSession(dataOffset: number, dataLength: number, sessionOptionsHandle: number): Promise<number>;
  _OrtReleaseSession(sessionHandle: number): void;
  _OrtGetInputOutputCount(sessionHandle: number, inputCountOffset: number, outputCountOffset: number): number;
  _OrtGetInputName(sessionHandle: number, index: number): number;
  _OrtGetOutputName(sessionHandle: number, index: number): number;

  _OrtFree(stringHandle: number): void;

  _OrtCreateTensor(
    dataType: number,
    dataOffset: number,
    dataLength: number,
    dimsOffset: number,
    dimsLength: number,
    dataLocation: number,
  ): number;
  _OrtGetTensorData(
    tensorHandle: number,
    dataType: number,
    dataOffset: number,
    dimsOffset: number,
    dimsLength: number,
  ): number;
  _OrtReleaseTensor(tensorHandle: number): void;
  _OrtCreateBinding(sessionHandle: number): number;
  _OrtBindInput(bindingHandle: number, nameOffset: number, tensorHandle: number): Promise<number>;
  _OrtBindOutput(bindingHandle: number, nameOffset: number, tensorHandle: number, location: number): number;
  _OrtClearBoundOutputs(ioBindingHandle: number): void;
  _OrtReleaseBinding(ioBindingHandle: number): void;
  _OrtRunWithBinding(
    sessionHandle: number,
    ioBindingHandle: number,
    outputCount: number,
    outputsOffset: number,
    runOptionsHandle: number,
  ): Promise<number>;
  _OrtRun(
    sessionHandle: number,
    inputNamesOffset: number,
    inputsOffset: number,
    inputCount: number,
    outputNamesOffset: number,
    outputCount: number,
    outputsOffset: number,
    runOptionsHandle: number,
  ): Promise<number>;

  _OrtCreateSessionOptions(
    graphOptimizationLevel: number,
    enableCpuMemArena: boolean,
    enableMemPattern: boolean,
    executionMode: number,
    enableProfiling: boolean,
    profileFilePrefix: number,
    logId: number,
    logSeverityLevel: number,
    logVerbosityLevel: number,
    optimizedModelFilePath: number,
  ): number;
  _OrtAppendExecutionProvider(sessionOptionsHandle: number, name: number): number;
  _OrtAddFreeDimensionOverride(sessionOptionsHandle: number, name: number, dim: number): number;
  _OrtAddSessionConfigEntry(sessionOptionsHandle: number, configKey: number, configValue: number): number;
  _OrtReleaseSessionOptions(sessionOptionsHandle: number): void;

  _OrtCreateRunOptions(logSeverityLevel: number, logVerbosityLevel: number, terminate: boolean, tag: number): number;
  _OrtAddRunConfigEntry(runOptionsHandle: number, configKey: number, configValue: number): number;
  _OrtReleaseRunOptions(runOptionsHandle: number): void;

  _OrtEndProfiling(sessionHandle: number): number;
}

/**
 * The interface of the WebAssembly module for ONNX Runtime, compiled from C++ source code by Emscripten.
 */
export interface OrtWasmModule extends EmscriptenModule, OrtInferenceAPIs, Partial<JSEP.Module> {
  // #region emscripten functions
  stackSave(): number;
  stackRestore(stack: number): void;
  stackAlloc(size: number): number;

  UTF8ToString(offset: number, maxBytesToRead?: number): string;
  lengthBytesUTF8(str: string): number;
  stringToUTF8(str: string, offset: number, maxBytes: number): void;
  // #endregion

  // #region config
  numThreads?: number;
  // #endregion
}
