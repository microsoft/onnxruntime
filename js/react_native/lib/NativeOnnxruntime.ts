// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// eslint-disable-next-line import/no-internal-modules
import type {TurboModule} from 'react-native/Libraries/TurboModule/RCTExport';
import {TurboModuleRegistry} from 'react-native';

// NOTE: These should be from InferenceSession in onnxruntime-common
// but currently got issue https://github.com/facebook/react-native/issues/36431
type SessionOptions = {};
type RunOptions = {};

/**
 * model loading information
 */
export type ModelLoadInfo = {
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
};

/**
 * JSIBlob is a blob object that exchange ArrayBuffer by OnnxruntimeJSIHelper.
 */
export type JSIBlob = {
  blobId: string; offset: number; size: number;
};

/**
 * Tensor type for react native, which doesn't allow ArrayBuffer in native bridge, so data will be stored as JSIBlob.
 */
export type EncodedTensor = {
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
};

export type Feeds = {
  [name: string]: EncodedTensor;
};

// SessionHanlder FetchesType is different from native module's one.
// It's because Java API doesn't support preallocated output values.
export type Fetches = string[];

export type Return = {
  [name: string]: EncodedTensor;
};

// NOTE: Currently we can't use types import from another files
// ref: https://github.com/facebook/react-native/issues/36431
export interface Spec extends TurboModule {
  loadModel(modelPath: string, options: SessionOptions): Promise<ModelLoadInfo>;
  loadModelFromBlob?(blob: JSIBlob, options: SessionOptions): Promise<ModelLoadInfo>;
  dispose(key: string): Promise<void>;
  run(key: string, feeds: Feeds, fetches: Fetches, options: RunOptions): Promise<Return>;
}

export default TurboModuleRegistry.get<Spec>('Onnxruntime');
