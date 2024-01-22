// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {MAX_CLIP, MIN_CLIP} from '../../util';

export interface InternalActivationAttributes {
  readonly activation: string;
  readonly clipMin?: number;
  readonly clipMax?: number;
  readonly alpha?: number;
  readonly beta?: number;
  readonly activationCacheKey: string;
}

export const getActivationSnippet = (attributes: InternalActivationAttributes, valueType: string):
    {activationFunction: string; applyActivation: string} => {
      switch (attributes.activation) {
        case 'Relu':
          return {activationFunction: '', applyActivation: `value = max(value, ${valueType}(0.0));`};
        case 'Sigmoid':
          return {
            activationFunction: '',
            applyActivation: `value = (${valueType}(1.0) / (${valueType}(1.0) + exp(-value)));`
          };
        case 'Clip':
          return {
            activationFunction: `const clip_min_=${valueType}(${attributes.clipMin!});const clip_max_=${valueType}(${
                attributes.clipMax!});`,
            applyActivation: 'value = clamp(value, clip_min_, clip_max_);'
          };
        case 'HardSigmoid':
          return {
            activationFunction:
                `const alpha_ = ${valueType}(${attributes.alpha!});const beta_ = ${valueType}(${attributes.beta!});`,
            applyActivation: `value = max(${valueType}(0.0), min(${valueType}(1.0), alpha_ * value + beta_));`
          };
          // TODO: adding other activations that can be fused.
        default:
          return {activationFunction: '', applyActivation: ''};
      }
    };

export const parseInternalActivationAttributes =
    (attributes: Record<string, unknown>|undefined): InternalActivationAttributes => {
      const activation = attributes?.activation as string || '';
      if (activation === 'HardSigmoid') {
        const [alpha, beta] = attributes?.activation_params as [number, number] || [0.2, 0.5];
        return {activation, alpha, beta, activationCacheKey: `${activation}:${alpha},${beta}`};
      }
      if (activation === 'Clip') {
        const [clipMin, clipMax] = attributes?.activation_params as [number, number] || [MIN_CLIP, MAX_CLIP];
        return {activation, clipMax, clipMin, activationCacheKey: `${activation}:${clipMin},${clipMax}`};
      }
      return {activation, activationCacheKey: activation};
    };
