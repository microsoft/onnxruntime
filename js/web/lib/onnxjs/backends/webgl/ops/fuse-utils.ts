// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Attribute} from '../../../attribute';
import {GlslValueFunction} from '../glsl-definitions';
import {glslClip, glslRelu, glslSigmoid} from './unary-op';

export interface InternalActivationAttributes {
  readonly activation: string;
  readonly clipMin: number;
  readonly clipMax: number;
}

export function getActicationSnippet(attributes: InternalActivationAttributes) {
  let func: GlslValueFunction;
  switch (attributes.activation) {
    case 'Relu':
      func = glslRelu();
      break;
    case 'Sigmoid':
      func = glslSigmoid();
      break;
    case 'Clip':
      func = glslClip(attributes.clipMin, attributes.clipMax);
      break;
    // TODO: adding other activations that can be fused.
    default:
      return {activationFunction: '', applyActivation: ''};
  }

  const activationName = func.name;
  const activationFunction = func.body;
  const applyActivation = `value = ${activationName}_(value);`;
  return {activationFunction, applyActivation};
}

export const parseInternalActivationAttributes = (attributes: Attribute): InternalActivationAttributes => ({
  activation: attributes.getString('__internal_activation', ''),
  clipMax: attributes.getFloat('__clip_max', 3.402823e+38),
  clipMin: attributes.getFloat('__clip_min', -3.402823e+38),
});
