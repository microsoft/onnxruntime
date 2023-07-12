// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// eslint-disable-next-line @typescript-eslint/no-unused-vars
import type {InferenceSession} from 'onnxruntime-common';
import {NativeModules} from 'react-native';
import Onnxruntime from './NativeOnnxruntime';
import type {ModelLoadInfo, EncodedTensor, Fetches, JSIBlob, Feeds, Return} from './NativeOnnxruntime';

/**
 * Binding exports a simple synchronized inference session object wrap.
 */
export declare namespace Binding {
  type ModelLoadInfoType = ModelLoadInfo;
  type EncodedTensorType = EncodedTensor;

  type SessionOptions = InferenceSession.SessionOptions;
  type RunOptions = InferenceSession.RunOptions;

  type FeedsType = Feeds;
  type FetchesType = Fetches;
  type ReturnType = Return;
  type JSIBlobType = JSIBlob;

  interface InferenceSession {
    loadModel(modelPath: string, options: SessionOptions): Promise<ModelLoadInfoType>;
    loadModelFromBlob?(blob: JSIBlobType, options: SessionOptions): Promise<ModelLoadInfoType>;
    dispose(key: string): Promise<void>;
    run(key: string, feeds: FeedsType, fetches: FetchesType, options: RunOptions): Promise<ReturnType>;
  }
}

// export native binding
const {OnnxruntimeJSIHelper} = NativeModules;
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
