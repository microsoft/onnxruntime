// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession} from 'onnxruntime-common';

import {allocWasmString, iterateExtraOptions} from './options-utils';
import {getInstance} from './wasm-factory';

const getGraphOptimzationLevel = (graphOptimizationLevel: string|unknown): number => {
  switch (graphOptimizationLevel) {
    case 'disabled':
      return 0;
    case 'basic':
      return 1;
    case 'extended':
      return 2;
    case 'all':
      return 99;
    default:
      throw new Error(`unsupported graph optimization level: ${graphOptimizationLevel}`);
  }
};

const getExecutionMode = (executionMode: 'sequential'|'parallel'): number => {
  switch (executionMode) {
    case 'sequential':
      return 0;
    case 'parallel':
      return 1;
    default:
      throw new Error(`unsupported execution mode: ${executionMode}`);
  }
};

export const setSessionOptions = (options?: InferenceSession.SessionOptions): [number, number[]] => {
  const wasm = getInstance();
  let sessionOptionsHandle = 0;
  const allocs: number[] = [];

  const sessionOptions: InferenceSession.SessionOptions = options || {};

  try {
    if (options === undefined || options.graphOptimizationLevel === undefined) {
      sessionOptions.graphOptimizationLevel = 'all';
    }
    const graphOptimizationLevel = getGraphOptimzationLevel(sessionOptions.graphOptimizationLevel!);

    if (options === undefined || options.enableCpuMemArena === undefined) {
      sessionOptions.enableCpuMemArena = true;
    }

    if (options === undefined || options.enableMemPattern === undefined) {
      sessionOptions.enableMemPattern = true;
    }

    if (options === undefined || options.executionMode === undefined) {
      sessionOptions.executionMode = 'sequential';
    }
    const executionMode = getExecutionMode(sessionOptions.executionMode!);

    let logIdDataOffset = 0;
    if (options !== undefined && options.logId !== undefined) {
      logIdDataOffset = allocWasmString(options.logId, allocs);
    }

    if (options === undefined || options.logSeverityLevel === undefined) {
      sessionOptions.logSeverityLevel = 2;  // Default to warning
    } else if (
        typeof options.logSeverityLevel !== 'number' || options.logSeverityLevel < 0 || options.logSeverityLevel > 4) {
      throw new Error(`log serverity level is not valid: ${options.logSeverityLevel}`);
    }

    if (options === undefined || options.logVerbosityLevel === undefined) {
      sessionOptions.logVerbosityLevel = 0;  // Default to 0
    } else if (typeof options.logVerbosityLevel !== 'number') {
      throw new Error(`log verbosity level is not valid: ${options.logVerbosityLevel}`);
    }

    // TODO: Support profiling
    sessionOptions.enableProfiling = false;

    sessionOptionsHandle = wasm._OrtCreateSessionOptions(
        graphOptimizationLevel, !!sessionOptions.enableCpuMemArena!, !!sessionOptions.enableMemPattern!, executionMode,
        sessionOptions.enableProfiling, 0, logIdDataOffset, sessionOptions.logSeverityLevel!,
        sessionOptions.logVerbosityLevel!);
    if (sessionOptionsHandle === 0) {
      throw new Error('Can\'t create session options');
    }

    if (options !== undefined && options.extra !== undefined) {
      iterateExtraOptions(options.extra, '', {
        handle: (key, value) => {
          const keyDataOffset = allocWasmString(key, allocs);
          const valueDataOffset = allocWasmString(value, allocs);

          if (wasm._OrtAddSessionConfigEntry(sessionOptionsHandle, keyDataOffset, valueDataOffset) !== 0) {
            throw new Error(`Can't set a session config entry: ${key} - ${value}`);
          }
        }
      });
    }

    return [sessionOptionsHandle, allocs];
  } catch (e) {
    if (sessionOptionsHandle !== 0) {
      wasm._OrtReleaseSessionOptions(sessionOptionsHandle);
    }
    allocs.forEach(wasm._free);
    throw e;
  }
};
