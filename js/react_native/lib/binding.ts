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
 * Tensor type for react native, which doesn't allow ArrayBuffer, so data will be encoded as Base64 string.
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
   * the Base64 encoded string of the buffer data of the tensor.
   * if data is string array, it won't be encoded as Base64 string.
   */
  readonly data: string|string[];
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
    run(key: string, feeds: FeedsType, fetches: FetchesType, options: RunOptions): Promise<ReturnType>;
  }
}

// export native binding
const {Onnxruntime} = NativeModules;
export const binding = Onnxruntime as Binding.InferenceSession;
