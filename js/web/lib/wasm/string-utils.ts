// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {getInstance} from './wasm-factory';

export const allocWasmString = (data: string, allocs: bigint[]): bigint => {
  const wasm = getInstance();

  const dataLength = wasm.lengthBytesUTF8(data) * 4;
  const dataOffset = wasm._malloc(BigInt(dataLength));
  wasm.stringToUTF8(data, Number(dataOffset), dataLength);
  allocs.push(dataOffset);

  return dataOffset;
};
