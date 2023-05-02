// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession} from 'onnxruntime-common';

import {iterateExtraOptions} from './options-utils';
import {allocWasmString} from './string-utils';
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

const appendDefaultOptions = (options: InferenceSession.SessionOptions): void => {
  if (!options.extra) {
    options.extra = {};
  }
  if (!options.extra.session) {
    options.extra.session = {};
  }
  const session = options.extra.session as Record<string, string>;
  if (!session.use_ort_model_bytes_directly) {
    // eslint-disable-next-line camelcase
    session.use_ort_model_bytes_directly = '1';
  }

  // if using JSEP with WebGPU, always disable memory pattern
  if (options.executionProviders &&
      options.executionProviders.some(ep => (typeof ep === 'string' ? ep : ep.name) === 'webgpu')) {
    options.enableMemPattern = false;
  }
};

const setExecutionProviders =
    (sessionOptionsHandle: number, executionProviders: readonly InferenceSession.ExecutionProviderConfig[],
     allocs: number[]): void => {
      for (const ep of executionProviders) {
        let epName = typeof ep === 'string' ? ep : ep.name;

        // check EP name
        switch (epName) {
          case 'xnnpack':
            epName = 'XNNPACK';
            break;
          case 'webgpu':
            epName = 'JS';
            break;
          case 'wasm':
          case 'cpu':
            continue;
          default:
            throw new Error(`not supported EP: ${epName}`);
        }

        const epNameDataOffset = allocWasmString(epName, allocs);
        if (getInstance()._OrtAppendExecutionProvider(sessionOptionsHandle, epNameDataOffset) !== 0) {
          throw new Error(`Can't append execution provider: ${epName}`);
        }
      }
    };

export const setSessionOptions = (options?: InferenceSession.SessionOptions): [number, number[]] => {
  const wasm = getInstance();
  let sessionOptionsHandle = 0;
  const allocs: number[] = [];

  const sessionOptions: InferenceSession.SessionOptions = options || {};
  appendDefaultOptions(sessionOptions);

  try {
    const graphOptimizationLevel = getGraphOptimzationLevel(sessionOptions.graphOptimizationLevel ?? 'all');
    const executionMode = getExecutionMode(sessionOptions.executionMode ?? 'sequential');
    const logIdDataOffset =
        typeof sessionOptions.logId === 'string' ? allocWasmString(sessionOptions.logId, allocs) : 0;

    const logSeverityLevel = sessionOptions.logSeverityLevel ?? 2;  // Default to 2 - warning
    if (!Number.isInteger(logSeverityLevel) || logSeverityLevel < 0 || logSeverityLevel > 4) {
      throw new Error(`log serverity level is not valid: ${logSeverityLevel}`);
    }

    const logVerbosityLevel = sessionOptions.logVerbosityLevel ?? 0;  // Default to 0 - verbose
    if (!Number.isInteger(logVerbosityLevel) || logVerbosityLevel < 0 || logVerbosityLevel > 4) {
      throw new Error(`log verbosity level is not valid: ${logVerbosityLevel}`);
    }

    const optimizedModelFilePathOffset = typeof sessionOptions.optimizedModelFilePath === 'string' ?
        allocWasmString(sessionOptions.optimizedModelFilePath, allocs) :
        0;

    sessionOptionsHandle = wasm._OrtCreateSessionOptions(
        graphOptimizationLevel, !!sessionOptions.enableCpuMemArena, !!sessionOptions.enableMemPattern, executionMode,
        !!sessionOptions.enableProfiling, 0, logIdDataOffset, logSeverityLevel, logVerbosityLevel,
        optimizedModelFilePathOffset);
    if (sessionOptionsHandle === 0) {
      throw new Error('Can\'t create session options');
    }

    if (sessionOptions.executionProviders) {
      setExecutionProviders(sessionOptionsHandle, sessionOptions.executionProviders, allocs);
    }

    if (sessionOptions.extra !== undefined) {
      iterateExtraOptions(sessionOptions.extra, '', new WeakSet<Record<string, unknown>>(), (key, value) => {
        const keyDataOffset = allocWasmString(key, allocs);
        const valueDataOffset = allocWasmString(value, allocs);

        if (wasm._OrtAddSessionConfigEntry(sessionOptionsHandle, keyDataOffset, valueDataOffset) !== 0) {
          throw new Error(`Can't set a session config entry: ${key} - ${value}`);
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
