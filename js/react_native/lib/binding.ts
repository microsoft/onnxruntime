// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// eslint-disable-next-line @typescript-eslint/no-unused-vars
import type {InferenceSession} from 'onnxruntime-common';
import {NativeModules} from 'react-native';

/**
 * model loading information
 */
interface ModelLoadInfo {
  /**
   * Key for an instance of InferenceSession, which is passed to run() function as parameter.
   */
  readonly key: string;

  /**
   * Get input names of the loaded model.
   */
  readonly inputNames: string[];

  /**
   * Get output names of the loaded model.
   */
  readonly outputNames: string[];
}

/**
 * JSIBlob is a blob object that exchange ArrayBuffer by OnnxruntimeJSIHelper.
 */
export type JSIBlob = {
  blobId: string; offset: number; size: number;
};

/**
 * Tensor type for react native, which doesn't allow ArrayBuffer in native bridge, so data will be stored as JSIBlob.
 */
interface EncodedTensor {
  /**
   * the dimensions of the tensor.
   */
  readonly dims: readonly number[];
  /**
   * the data type of the tensor.
   */
  readonly type: string;
  /**
   * the JSIBlob object of the buffer data of the tensor.
   * if data is string array, it won't be stored as JSIBlob.
   */
  readonly data: JSIBlob|string[];
}

/**
 * Binding exports a simple synchronized inference session object wrap.
 */
export declare namespace Binding {
  type ModelLoadInfoType = ModelLoadInfo;
  type EncodedTensorType = EncodedTensor;

  type SessionOptions = InferenceSession.SessionOptions;
  type RunOptions = InferenceSession.RunOptions;

  type FeedsType = {[name: string]: EncodedTensor};

  // SessionHanlder FetchesType is different from native module's one.
  // It's because Java API doesn't support preallocated output values.
  type FetchesType = string[];

  type ReturnType = {[name: string]: EncodedTensor};

  interface InferenceSession {
    loadModel(modelPath: string, options: SessionOptions): Promise<ModelLoadInfoType>;
    loadModelFromBlob?(blob: JSIBlob, options: SessionOptions): Promise<ModelLoadInfoType>;
    dispose(key: string): Promise<void>;
    run(key: string, feeds: FeedsType, fetches: FetchesType, options: RunOptions): Promise<ReturnType>;
  }
}

// export native binding
const {Onnxruntime, OnnxruntimeJSIHelper} = NativeModules;
export const binding = Onnxruntime as Binding.InferenceSession;

// install JSI helper global functions
OnnxruntimeJSIHelper.install();

declare global {
  // eslint-disable-next-line no-var
  var jsiOnnxruntimeStoreArrayBuffer: ((buffer: ArrayBuffer) => JSIBlob)|undefined;
  // eslint-disable-next-line no-var
  var jsiOnnxruntimeResolveArrayBuffer: ((blob: JSIBlob) => ArrayBuffer)|undefined;
}

export const jsiHelper = {
  storeArrayBuffer: globalThis.jsiOnnxruntimeStoreArrayBuffer || (() => {
                      throw new Error(
                          'jsiOnnxruntimeStoreArrayBuffer is not found, ' +
                          'please make sure OnnxruntimeJSIHelper installation is successful.');
                    }),
  resolveArrayBuffer: globalThis.jsiOnnxruntimeResolveArrayBuffer || (() => {
                        throw new Error(
                            'jsiOnnxruntimeResolveArrayBuffer is not found, ' +
                            'please make sure OnnxruntimeJSIHelper installation is successful.');
                      }),
};

// Remove global functions after installation
{
  delete globalThis.jsiOnnxruntimeStoreArrayBuffer;
  delete globalThis.jsiOnnxruntimeResolveArrayBuffer;
}
