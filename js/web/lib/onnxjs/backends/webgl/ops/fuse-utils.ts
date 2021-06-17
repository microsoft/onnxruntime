// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
    default:
      // TODO: adding other activations that can be fused.
      activationName = '';
      activationFunction = '';
  }
  const applyActivation = activation ? `
  value = ${activationName}(value);` :
                                       '';
  return {activationFunction, applyActivation};
}
