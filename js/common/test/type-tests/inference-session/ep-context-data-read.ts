// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as ort from 'onnxruntime-common';

const validOptions: ort.InferenceSession.SessionOptions = {
  epContextDataRead: {
    callback: (name) => new TextEncoder().encode(name),
    maxDataSize: 1024,
  },
};
void validOptions;

const invalidCallback: ort.InferenceSession.SessionOptions = {
  epContextDataRead: {
    // {type-tests}|fail|1|2740
    callback: async () => new Uint8Array(),
    maxDataSize: 1024,
  },
};
void invalidCallback;

const invalidReturnType: ort.InferenceSession.SessionOptions = {
  epContextDataRead: {
    // {type-tests}|fail|1|2740
    callback: () => new ArrayBuffer(0),
    maxDataSize: 1024,
  },
};
void invalidReturnType;
