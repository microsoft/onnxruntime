// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Attribute} from '../../../attribute';
import {MAX_CLIP, MIN_CLIP} from '../../../util';

export interface InternalActivationAttributes {
  readonly activation: string;
  readonly clipMin?: number;
  readonly clipMax?: number;
  readonly activationCacheKey: string;
}

export function getActicationSnippet(attributes: InternalActivationAttributes) {
  switch (attributes.activation) {
    case 'Relu':
      return {activationFunction: '', applyActivation: 'value = max(value, 0.0);'};
    case 'Sigmoid':
      return {activationFunction: '', applyActivation: 'value = (1.0 / (1.0 + exp(-value)));'};
    case 'Clip':
      return {
        activationFunction: `let clip_min_=f32(${attributes.clipMin!});let clip_max_=f32(${attributes.clipMax!});`,
        applyActivation: 'value = clamp(value, clip_min_, clip_max_);'
      };
      // TODO: adding other activations that can be fused.
    default:
      return {activationFunction: '', applyActivation: ''};
  }
}

export const parseInternalActivationAttributes = (attributes: Attribute): InternalActivationAttributes => {
  const activation = attributes.getString('activation', '');

  if (activation === 'Clip') {
    const [clipMin, clipMax] = attributes.getFloats('activation_params', [MIN_CLIP, MAX_CLIP]);
    return {activation, clipMax, clipMin, activationCacheKey: `${activation}:${clipMin},${clipMax}`};
  }
  return {activation, activationCacheKey: activation};
};
