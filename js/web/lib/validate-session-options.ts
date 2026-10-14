// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import type { InferenceSession } from 'onnxruntime-common';

export const validateSessionOptions = (options?: InferenceSession.SessionOptions): void => {
  if (options?.epContextDataRead) {
    throw new Error('session option "epContextDataRead" is not supported by ONNX Runtime Web');
  }
};
