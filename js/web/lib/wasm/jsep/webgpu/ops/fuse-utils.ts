// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {MAX_CLIP, MIN_CLIP} from '../../util';

export interface InternalActivationAttributes {
  readonly activation: string;
  readonly clipMin?: number;
  readonly clipMax?: number;
}

export const getActivationSnippet = (attributes: InternalActivationAttributes, valueType: string): string => {
  switch (attributes.activation) {
    case 'Relu':
      return `value = max(value, ${valueType}(0.0));`;
    case 'Sigmoid':
      return `value = (${valueType}(1.0) / (${valueType}(1.0) + exp(-value)));`;
    case 'Clip':
      return `value = clamp(value, ${valueType}(uniforms.clip_min), ${valueType}(uniforms.clip_max));`;
    // TODO: adding other activations that can be fused.
    default:
      return '';
  }
};

export const parseInternalActivationAttributes =
    (attributes: Record<string, unknown>|undefined): InternalActivationAttributes => {
      const activation = attributes?.activation as string || '';

      if (activation === 'Clip') {
        const [clipMin, clipMax] = attributes?.activation_params as [number, number] || [MIN_CLIP, MAX_CLIP];
        return {activation, clipMax, clipMin};
      }
      return {activation};
    };
