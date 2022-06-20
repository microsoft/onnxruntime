// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession} from 'onnxruntime-common';

import {iterateExtraOptions} from './options-utils';
import {allocWasmString} from './string-utils';
import {getInstance} from './wasm-factory';

export const setRunOptions = (options: InferenceSession.RunOptions): [number, number[]] => {
  const wasm = getInstance();
  let runOptionsHandle = 0;
  const allocs: number[] = [];

  const runOptions: InferenceSession.RunOptions = options || {};

  try {
    if (options?.logSeverityLevel === undefined) {
      runOptions.logSeverityLevel = 2;  // Default to warning
    } else if (
        typeof options.logSeverityLevel !== 'number' || !Number.isInteger(options.logSeverityLevel) ||
        options.logSeverityLevel < 0 || options.logSeverityLevel > 4) {
      throw new Error(`log serverity level is not valid: ${options.logSeverityLevel}`);
    }

    if (options?.logVerbosityLevel === undefined) {
      runOptions.logVerbosityLevel = 0;  // Default to 0
    } else if (typeof options.logVerbosityLevel !== 'number' || !Number.isInteger(options.logVerbosityLevel)) {
      throw new Error(`log verbosity level is not valid: ${options.logVerbosityLevel}`);
    }

    if (options?.terminate === undefined) {
      runOptions.terminate = false;
    }

    let tagDataOffset = 0;
    if (options?.tag !== undefined) {
      tagDataOffset = allocWasmString(options.tag, allocs);
    }

    runOptionsHandle = wasm._OrtCreateRunOptions(
        runOptions.logSeverityLevel!, runOptions.logVerbosityLevel!, !!runOptions.terminate!, tagDataOffset);
    if (runOptionsHandle === 0) {
      throw new Error('Can\'t create run options');
    }

    if (options?.extra !== undefined) {
      iterateExtraOptions(options.extra, '', new WeakSet<Record<string, unknown>>(), (key, value) => {
        const keyDataOffset = allocWasmString(key, allocs);
        const valueDataOffset = allocWasmString(value, allocs);

        if (wasm._OrtAddRunConfigEntry(runOptionsHandle, keyDataOffset, valueDataOffset) !== 0) {
          throw new Error(`Can't set a run config entry: ${key} - ${value}`);
        }
      });
    }

    return [runOptionsHandle, allocs];
  } catch (e) {
    if (runOptionsHandle !== 0) {
      wasm._OrtReleaseRunOptions(runOptionsHandle);
    }
    allocs.forEach(wasm._free);
    throw e;
  }
};
