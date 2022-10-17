// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {OrtWasmModule} from '../binding/ort-wasm';

import {WebGpuBackend} from './backend-webgpu';

export const init = async(module: OrtWasmModule): Promise<void> => {
  // init JSEP if available
  const init = module.jsepInit;
  if (init) {
    const backend = new WebGpuBackend();
    await backend.initialize();

    init(
        // backend
        {backend},

        // jsepAlloc()
        (size: number) => {
          // eslint-disable-next-line no-console
          console.log(`jsepAlloc: ${size}`);
          return backend.alloc(size);
        },

        // jsepFree()
        (ptr: number) => {
          // eslint-disable-next-line no-console
          console.log(`jsepFree: ${ptr}`);
          return backend.free(ptr);
        },

        // jsepUpload(src, dst, size)
        (dataOffset: number, gpuDataId: number, size: number) => {
          // eslint-disable-next-line no-console
          console.log('jsepUpload');
          const data = module.HEAPU8.subarray(dataOffset, dataOffset + size);
          backend.upload(dataOffset, data, gpuDataId);
        },
        (_src: number, _dst: number) => {
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
