// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession as InferenceSessionInterface} from './inference-session';
import {OnnxValue} from './onnx-value';

/**
 * Binding exports a simple synchronized inference session object wrap.
 */
export declare namespace Binding {
  export interface InferenceSession {
    loadModel(modelPath: string, options: InferenceSessionInterface.SessionOptions): void;
    loadModel(
        buffer: ArrayBuffer, byteOffset: number, byteLength: number,
        options: InferenceSessionInterface.SessionOptions): void;

    readonly inputNames: string[];
    readonly outputNames: string[];

    run(feeds: InferenceSession.FeedsType, fetches: InferenceSession.FetchesType,
        options: InferenceSession.RunOptions): InferenceSession.ReturnType;
  }

  export namespace InferenceSession {
    type FeedsType = {[name: string]: OnnxValue};
    type FetchesType = {[name: string]: OnnxValue | null};
    type ReturnType = {[name: string]: OnnxValue};
    type RunOptions = InferenceSessionInterface.RunOptions;
  }

  export interface InferenceSessionConstructor {
    new(): InferenceSession;
  }
}

// export native binding
export const binding =
    // eslint-disable-next-line @typescript-eslint/no-require-imports, import/no-internal-modules
    require(`../bin/napi-v3/${process.platform}/${process.arch}/onnxruntime_binding.node`) as
    {InferenceSession: Binding.InferenceSessionConstructor};
