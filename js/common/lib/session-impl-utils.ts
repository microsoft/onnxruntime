// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import {InferenceSession as InferenceSession} from './inference-session.js';

export const processModel = (model: string|ArrayBufferLike|Uint8Array): string|Uint8Array => {
  if (typeof model === 'string' || model instanceof Uint8Array) {
    return model;
  } else if (model instanceof ArrayBuffer) {
    return new Uint8Array(model);
  } else {
    throw new TypeError('Unexpected argument: must be \'path\' or \'buffer\'.');
  }
};

export const processModelOrOptions =
    (modelOrOptions: string|ArrayBufferLike|Uint8Array|InferenceSession.SessionOptions,
     options_placeholder: InferenceSession.SessionOptions, model_placeholder: string|Uint8Array):
        [options_placeholder: InferenceSession.SessionOptions, model_placeholder: string|Uint8Array] => {
          if (typeof modelOrOptions === 'string' || modelOrOptions instanceof Uint8Array) {
            model_placeholder = modelOrOptions;
          } else if (modelOrOptions instanceof ArrayBuffer) {
            model_placeholder = new Uint8Array(modelOrOptions);
          } else if (typeof modelOrOptions === 'object') {
            options_placeholder = modelOrOptions as InferenceSession.SessionOptions;
          } else {
            throw new TypeError('Unexpected argument: must be \'path\', \'buffer\', or \'SessionOptions\'.');
          }
          return [options_placeholder, model_placeholder];
        };
