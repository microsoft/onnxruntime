// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {getInstance} from './wasm-factory';

interface ExtraOptionsHandler {
  (name: string, value: string): void;
}

export const iterateExtraOptions =
    (options: Record<string, unknown>, prefix: string, seen: WeakSet<Record<string, unknown>>,
     handler: ExtraOptionsHandler): void => {
      if (typeof options == 'object' && options !== null) {
        if (seen.has(options)) {
          throw new Error('Circular reference in options');
        } else {
          seen.add(options);
        }
      }

      Object.entries(options).forEach(([key, value]) => {
        const name = (prefix) ? prefix + key : key;
        if (typeof value === 'object') {
          iterateExtraOptions(value as Record<string, unknown>, name + '.', seen, handler);
        } else if (typeof value === 'string' || typeof value === 'number') {
          handler(name, value.toString());
        } else if (typeof value === 'boolean') {
          handler(name, (value) ? '1' : '0');
        } else {
          throw new Error(`Can't handle extra config type: ${typeof value}`);
        }
      });
    };

export const allocWasmString = (data: string, allocs: number[]): number => {
  const wasm = getInstance();

  const dataLength = wasm.lengthBytesUTF8(data) + 1;
  const dataOffset = wasm._malloc(dataLength);
  wasm.stringToUTF8(data, dataOffset, dataLength);
  allocs.push(dataOffset);

  return dataOffset;
};
