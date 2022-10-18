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
        (size: number) => backend.alloc(size),

        // jsepFree()
        (ptr: number) => backend.free(ptr),

        // jsepUpload(src, dst, size)
        (dataOffset: number, gpuDataId: number, size: number) => {
          // eslint-disable-next-line no-console
          console.log('jsepUpload');
          const data = module.HEAPU8.subarray(dataOffset, dataOffset + size);
          backend.upload(gpuDataId, data);
        },

        // jsepDownload(src, dst, size)
        async(gpuDataId: number, dataOffset: number, size: number):
            Promise<void> => {
              // eslint-disable-next-line no-console
              console.log('jsepDownload');

              const data = module.HEAPU8.subarray(dataOffset, dataOffset + size);
              await backend.download(gpuDataId, data);
            },

        (contextDataOffset: number, output: (index: number) => number) => {
          // eslint-disable-next-line no-console
          console.log('jsepRun');
          return 42;
        });
  }
};
