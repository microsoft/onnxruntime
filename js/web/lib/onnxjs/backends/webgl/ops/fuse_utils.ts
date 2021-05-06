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
      activationName = '';
      activationFunction = '';
  }
  const applyActivation = activation ? `
  value = ${activationName}(value);` :
                                       '';
  return {activationFunction, applyActivation};
}
