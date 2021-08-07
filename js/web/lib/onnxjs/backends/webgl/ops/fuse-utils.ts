// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {glslClip} from './clip';
import {glslRelu, glslSigmoid} from './unary-op';

export function getActicationSnippet(activation: string) {
  let activationFunction = '';
  let activationName = '';
  switch (activation) {
    case 'Relu':
      activationName = glslRelu().name;
      activationFunction = glslRelu().body;
      break;
    case 'Sigmoid':
      activationName = glslSigmoid().name;
      activationFunction = glslSigmoid().body;
      break;
    case 'Clip':
      activationName = glslClip().name;
      activationFunction = glslClip().body;
      break;
    default:
      // TODO: adding other activations that can be fused.
      activationName = '';
      activationFunction = '';
  }

  const applyActivation = activation ? (activation === 'Clip' ? `
  value = ${activationName}(value, max, min);` :
                                                                `
  value = ${activationName}(value);`) :
                                       '';
  return {activationFunction, applyActivation};
}
