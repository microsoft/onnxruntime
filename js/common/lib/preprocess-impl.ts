// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Preprocessor} from './preprocess.js';

const registeredPreprocessors: Map<string, Map<string, () => Promise<Preprocessor>>> = new Map();

/**
 * Register a preprocessor.
 *
 * A preprocessor can process a certain type of raw data into a data type that can be directly used as feeds by the
 * backend for inference.
 *
 * @param backendName - the backend name.
 * @param input - the input type. usually a raw data type.
 * @param output - the output data type.
 * @param getPreprocessor - a function that returns a promise of a preprocessor.
 *
 * @ignore
 */
export const registerPreprocessor =
    (backendName: string, input: string, output: string, getPreprocessor: () => Promise<Preprocessor>): void => {
      let preprocessors = registeredPreprocessors.get(backendName);
      if (preprocessors === undefined) {
        preprocessors = new Map();
        registeredPreprocessors.set(backendName, preprocessors);
      }
      preprocessors.set(`${input}:${output}`, getPreprocessor);
    };

/**
 * Resolve a preprocessor.
 *
 * @param backendName - the backend name.
 * @param input - the input type. usually a raw data type.
 * @param output - the output data type.
 *
 * @ignore
 */
export const resolvePreprocessor =
    async(backendName: string, input: string, output: string): Promise<Preprocessor|undefined> => {
  const preprocessors = registeredPreprocessors.get(backendName);
  if (preprocessors === undefined) {
    return undefined;
  }
  const getPreprocessor = preprocessors.get(`${input}:${output}`);
  if (getPreprocessor === undefined) {
    return undefined;
  }
  return getPreprocessor();
};
