// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as ort from 'onnxruntime-web';

export const print = () => {
  console.log('onnxruntime-web version:', ort.env.versions);
};
