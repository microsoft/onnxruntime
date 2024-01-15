// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession} from 'onnxruntime-common';

import {getInstance} from './wasm-factory';
import {allocWasmString, checkLastError, iterateExtraOptions} from './wasm-utils';

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
          case 'webnn':
            epName = 'WEBNN';
            if (typeof ep !== 'string') {
              const webnnOptions = ep as InferenceSession.WebNNExecutionProviderOption;
              if (webnnOptions?.deviceType) {
                const keyDataOffset = allocWasmString('deviceType', allocs);
                const valueDataOffset = allocWasmString(webnnOptions.deviceType, allocs);
                if (getInstance()._OrtAddSessionConfigEntry(sessionOptionsHandle, keyDataOffset, valueDataOffset) !==
                    0) {
                  checkLastError(`Can't set a session config entry: 'deviceType' - ${webnnOptions.deviceType}.`);
                }
              }
              if (webnnOptions?.numThreads) {
                let numThreads = webnnOptions.numThreads;
                // Just ignore invalid webnnOptions.numThreads.
                if (typeof numThreads != 'number' || !Number.isInteger(numThreads) || numThreads < 0) {
                  numThreads = 0;
                }
                const keyDataOffset = allocWasmString('numThreads', allocs);
                const valueDataOffset = allocWasmString(numThreads.toString(), allocs);
                if (getInstance()._OrtAddSessionConfigEntry(sessionOptionsHandle, keyDataOffset, valueDataOffset) !==
                    0) {
                  checkLastError(`Can't set a session config entry: 'numThreads' - ${webnnOptions.numThreads}.`);
                }
              }
              if (webnnOptions?.powerPreference) {
                const keyDataOffset = allocWasmString('powerPreference', allocs);
                const valueDataOffset = allocWasmString(webnnOptions.powerPreference, allocs);
                if (getInstance()._OrtAddSessionConfigEntry(sessionOptionsHandle, keyDataOffset, valueDataOffset) !==
                    0) {
                  checkLastError(
                      `Can't set a session config entry: 'powerPreference' - ${webnnOptions.powerPreference}.`);
                }
              }
            }
            break;
          case 'webgpu':
            epName = 'JS';
            if (typeof ep !== 'string') {
              const webgpuOptions = ep as InferenceSession.WebGpuExecutionProviderOption;
              if (webgpuOptions?.preferredLayout) {
                if (webgpuOptions.preferredLayout !== 'NCHW' && webgpuOptions.preferredLayout !== 'NHWC') {
                  throw new Error(`preferredLayout must be either 'NCHW' or 'NHWC': ${webgpuOptions.preferredLayout}`);
                }
                const keyDataOffset = allocWasmString('preferredLayout', allocs);
                const valueDataOffset = allocWasmString(webgpuOptions.preferredLayout, allocs);
                if (getInstance()._OrtAddSessionConfigEntry(sessionOptionsHandle, keyDataOffset, valueDataOffset) !==
                    0) {
                  checkLastError(
                      `Can't set a session config entry: 'preferredLayout' - ${webgpuOptions.preferredLayout}.`);
                }
              }
            }
            break;
          case 'wasm':
          case 'cpu':
            continue;
          default:
            throw new Error(`not supported execution provider: ${epName}`);
        }

        const epNameDataOffset = allocWasmString(epName, allocs);
        if (getInstance()._OrtAppendExecutionProvider(sessionOptionsHandle, epNameDataOffset) !== 0) {
          checkLastError(`Can't append execution provider: ${epName}.`);
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
      checkLastError('Can\'t create session options.');
    }

    if (sessionOptions.executionProviders) {
      setExecutionProviders(sessionOptionsHandle, sessionOptions.executionProviders, allocs);
    }

    if (sessionOptions.freeDimensionOverrides) {
      for (const [name, value] of Object.entries(sessionOptions.freeDimensionOverrides)) {
        if (typeof name !== 'string') {
          throw new Error(`free dimension override name must be a string: ${name}`);
        }
        if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
          throw new Error(`free dimension override value must be a non-negative integer: ${value}`);
        }
        const nameOffset = allocWasmString(name, allocs);
        if (wasm._OrtAddFreeDimensionOverride(sessionOptionsHandle, nameOffset, value) !== 0) {
          checkLastError(`Can't set a free dimension override: ${name} - ${value}.`);
        }
      }
    }

    if (sessionOptions.extra !== undefined) {
      iterateExtraOptions(sessionOptions.extra, '', new WeakSet<Record<string, unknown>>(), (key, value) => {
        const keyDataOffset = allocWasmString(key, allocs);
        const valueDataOffset = allocWasmString(value, allocs);

        if (wasm._OrtAddSessionConfigEntry(sessionOptionsHandle, keyDataOffset, valueDataOffset) !== 0) {
          checkLastError(`Can't set a session config entry: ${key} - ${value}.`);
        }
      });
    }

    return [sessionOptionsHandle, allocs];
  } catch (e) {
    if (sessionOptionsHandle !== 0) {
      wasm._OrtReleaseSessionOptions(sessionOptionsHandle);
    }
    allocs.forEach(alloc => wasm._free(alloc));
    throw e;
  }
};
