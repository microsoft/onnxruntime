// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { env } from 'onnxruntime-common';

export const createDeprecationWarning = (message: string): (() => void) => {
  let warned = false;

  return () => {
    if (warned || env.logLevel === 'error' || env.logLevel === 'fatal') {
      return;
    }

    warned = true;
    // eslint-disable-next-line no-console
    console.warn(message);
  };
};
