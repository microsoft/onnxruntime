// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {OrtWasmModule} from './binding/ort-wasm';

export const init = (module: OrtWasmModule): void => {
  // init JSEP if available
  const init = module.jsepInit;
  if (init) {
    init(
        {},
        (size: number) => {
          // eslint-disable-next-line no-console
          console.log(`jsepAlloc: ${size}`);
          return 1234;
        },
        (ptr: number) => {
          // eslint-disable-next-line no-console
          console.log(`jsepFree: ${ptr}`);
          return 5678;
        },
        (_a: number) => {
          // eslint-disable-next-line no-console
          console.log('jsepUpload');
          return 40;
        },
        (_a: number) => {
          // eslint-disable-next-line no-console
          console.log('jsepDownload');
          return 41;
        },
        (_a: number) => {
          // eslint-disable-next-line no-console
          console.log('jsepRun');
          return 42;
        });
  }
};
