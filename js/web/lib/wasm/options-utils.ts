// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {getInstance} from './wasm-factory';

interface ExtraOptionsHandler {
  handle(name: string, value: string): void;
  seen?: WeakSet<Record<string, unknown>>;
}

export const iterateExtraOptions =
    (options: Record<string, unknown>, prefix: string, handler: ExtraOptionsHandler): void => {
      if (handler.seen === undefined) {
        handler.seen = new WeakSet<Record<string, unknown>>();
      }
      if (typeof options == 'object' && options !== null) {
        if (handler.seen.has(options)) {
          throw new Error('Circular reference in options');
        } else {
          handler.seen.add(options);
        }
      }

      Object.entries(options).forEach(([key, value]) => {
        const name = (prefix) ? prefix + key : key;
        if (typeof value === 'object') {
          iterateExtraOptions(value as Record<string, unknown>, name + '.', handler);
        } else if (typeof value === 'string' || typeof value === 'number') {
          handler.handle(name, value.toString());
        } else if (typeof value === 'boolean') {
          handler.handle(name, (value) ? '1' : '0');
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
