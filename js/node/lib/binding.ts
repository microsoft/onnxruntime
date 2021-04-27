// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession, OnnxValue} from 'onnxruntime-common';

type SessionOptions = InferenceSession.SessionOptions;
type FeedsType = {
  [name: string]: OnnxValue;
};
type FetchesType = {
  [name: string]: OnnxValue|null;
};
type ReturnType = {
  [name: string]: OnnxValue;
};
type RunOptions = InferenceSession.RunOptions;


/**
 * Binding exports a simple synchronized inference session object wrap.
 */
export declare namespace Binding {
  export interface InferenceSession {
    loadModel(modelPath: string, options: SessionOptions): void;
    loadModel(buffer: ArrayBuffer, byteOffset: number, byteLength: number, options: SessionOptions): void;

    readonly inputNames: string[];
    readonly outputNames: string[];

    run(feeds: FeedsType, fetches: FetchesType, options: RunOptions): ReturnType;
  }

  export interface InferenceSessionConstructor {
    new(): InferenceSession;
  }
}

// export native binding
export const binding =
    // eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-var-requires
    require(`../bin/napi-v3/${process.platform}/${process.arch}/onnxruntime_binding.node`) as
    // eslint-disable-next-line @typescript-eslint/naming-convention
    {InferenceSession: Binding.InferenceSessionConstructor};
