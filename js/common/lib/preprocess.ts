// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from './tensor.js';

/* eslint-disable @typescript-eslint/naming-convention */
export interface PreprocessorDataInputTypeMapping {
  'image-data': ImageData;
  'image-bitmap': ImageBitmap;
  'image-url': string;
  'image-element': HTMLImageElement;
}

interface PreprocessorDataOutputForGpuBuffer {
  readonly gpuBuffer: Tensor['gpuBuffer'];
  readonly dispose?: Tensor['dispose'];
}


export interface PreprocessorDataOutputTypeMapping {
  'cpu': Tensor['data'];
  'gpu-buffer': PreprocessorDataOutputForGpuBuffer;
}
/* eslint-enable @typescript-eslint/naming-convention */

/**
 * A preprocess plan describes how to preprocess a certain type of raw data into an inference-ready data.
 */
export interface PreprocessPlan<
    Input extends keyof PreprocessorDataInputTypeMapping = keyof PreprocessorDataInputTypeMapping,
                  Output extends keyof PreprocessorDataOutputTypeMapping = keyof PreprocessorDataOutputTypeMapping> {
  data: PreprocessorDataInputTypeMapping[Input];
  input: Input;
  output: Output;
  options?: unknown;
}

export interface Preprocessor<
    Input extends keyof PreprocessorDataInputTypeMapping = keyof PreprocessorDataInputTypeMapping,
                  Output extends keyof PreprocessorDataOutputTypeMapping = keyof PreprocessorDataOutputTypeMapping> {
  process(data: PreprocessorDataInputTypeMapping[Input], options?: unknown):
      Promise<PreprocessorDataOutputTypeMapping[Output]>;
}

export {registerPreprocessor} from './preprocess-impl.js';
