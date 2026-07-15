// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { NativeModules } from 'react-native';
import type { OrtApi as OrtApiType } from './api';

export const Module = NativeModules.Onnxruntime;

declare global {
  var OrtApi: OrtApiType;
}

if (typeof globalThis.OrtApi === 'undefined') {
  Module.install();
}

export const OrtApi =
  globalThis.OrtApi ??
  new Proxy(
    {},
    {
      get: () => {
        throw new Error('OrtApi is not initialized. Please make sure Onnxruntime installation is successful.');
      },
    },
  );
