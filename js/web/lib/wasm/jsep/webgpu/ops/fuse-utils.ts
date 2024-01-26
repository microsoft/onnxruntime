// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {MAX_CLIP, MIN_CLIP} from '../../util';
import {ProgramUniform} from '../types';

import {UniformsArrayType} from './common';

export interface InternalActivationAttributes {
  readonly activation: string;
  readonly clipMin?: number;
  readonly clipMax?: number;
  readonly alpha?: number;
  readonly beta?: number;
}

export const getActivationSnippet = (attributes: InternalActivationAttributes, valueType: string): string => {
  switch (attributes.activation) {
    case 'Relu':
      return `value = max(value, ${valueType}(0.0));`;
    case 'Sigmoid':
      return `value = (${valueType}(1.0) / (${valueType}(1.0) + exp(-value)));`;
    case 'Clip':
      return `value = clamp(value, ${valueType}(uniforms.clip_min), ${valueType}(uniforms.clip_max));`;
    case 'HardSigmoid':
      return `value = max(${valueType}(0.0), min(${valueType}(1.0), ${valueType}(uniforms.alpha) * value + ${
          valueType}(uniforms.beta)));`;
    // TODO: adding other activations that can be fused.
    default:
      return '';
  }
};

export const appendActivationUniformsData =
    (attributes: InternalActivationAttributes, programUniform: ProgramUniform[]) => {
      if (attributes.activation === 'Clip') {
        programUniform.push({type: 'float32', data: attributes.clipMax!}, {type: 'float32', data: attributes.clipMin!});
      } else if (attributes.activation === 'HardSigmoid') {
        programUniform.push({type: 'float32', data: attributes.alpha!}, {type: 'float32', data: attributes.beta!});
      }
    };

export const appendActivationUniforms = (attributes: InternalActivationAttributes, uniforms: UniformsArrayType) => {
  if (attributes.activation === 'Clip') {
    uniforms.push({name: 'clip_max', type: 'f32'}, {name: 'clip_min', type: 'f32'});
  } else if (attributes.activation === 'HardSigmoid') {
    uniforms.push({name: 'alpha', type: 'f32'}, {name: 'beta', type: 'f32'});
  }
};

export const parseInternalActivationAttributes =
    (attributes: Record<string, unknown>|undefined): InternalActivationAttributes => {
      const activation = attributes?.activation as string || '';
      if (activation === 'HardSigmoid') {
        const [alpha, beta] = attributes?.activation_params as [number, number] || [0.2, 0.5];
        return {activation, alpha, beta};
      }
      if (activation === 'Clip') {
        const [clipMin, clipMax] = attributes?.activation_params as [number, number] || [MIN_CLIP, MAX_CLIP];
        return {activation, clipMax, clipMin};
      }
      return {activation};
    };
