// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Graph} from '../../../graph';
import {OperatorImplementation, OperatorInitialization} from '../../../operators';
import {Tensor} from '../../../tensor';
import {BroadcastUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, TextureType} from '../types';
import {getActicationSnippet, InternalActivationAttributes, parseInternalActivationAttributes} from './fuse-utils';
import {createPackedMatmulProgramInfo} from './matmul-pack';

export const matMul: OperatorImplementation<InternalActivationAttributes> =
    (inferenceHandler: WebGLInferenceHandler, inputs: Tensor[], attributes: InternalActivationAttributes): Tensor[] => {
      validateInputs(inputs);

      if (inferenceHandler.session.pack) {
        return [inferenceHandler.run(createPackedMatmulProgramInfo(inferenceHandler, inputs, attributes), inputs)];
      } else {
        return [inferenceHandler.run(createMatmulProgramInfo(inputs, attributes), inputs)];
      }
    };

export const parseMatMulAttributes: OperatorInitialization<InternalActivationAttributes> =
    (node: Graph.Node): InternalActivationAttributes => parseInternalActivationAttributes(node.attributes);

function createMatmulProgramInfo(inputs: Tensor[], activationAttributes: InternalActivationAttributes): ProgramInfo {
  const aShape = inputs[0].dims;
  const bShape = inputs[1].dims;
  const outputShape = BroadcastUtil.calcShape(aShape, bShape, true);
  if (!outputShape) {
    throw new Error('Can\'t use matmul on the given tensors');
  }
  const {activationFunction, applyActivation} = getActicationSnippet(activationAttributes);
  const rank = outputShape.length;
  const arank = aShape.length;
  const brank = bShape.length;
  const sharedDim = aShape[aShape.length - 1];
  const shaderSource = `
    ${activationFunction}
    float process(int indices[${rank}]) {
        int a[${arank}];
        int b[${brank}];
        bcastMatmulIndices_A(indices, a);
        bcastMatmulIndices_B(indices, b);

        float value;
        for (int k=0; k<${sharedDim}; ++k) {
            a[${arank - 1}] = k;
            b[${brank - 2}] = k;
            value += _A(a) * _B(b);
        }
        ${applyActivation}
        return value;
    }`;
  return {
    name: 'MatMul',
    inputTypes: [TextureType.unpacked, TextureType.unpacked],
    inputNames: ['A', 'B'],
    output: {dims: outputShape, type: inputs[0].type, textureType: TextureType.unpacked},
    shaderSource,
  };
}

const validateInputs = (inputs: Tensor[]): void => {
  if (!inputs || inputs.length !== 2) {
    throw new Error('MatMul requires 2 inputs.');
  }

  if (inputs[0].dims[inputs[0].dims.length - 1] !== inputs[1].dims[inputs[1].dims.length - 2]) {
    throw new Error('shared dimension does not match.');
  }

  if ((inputs[0].type !== 'float32' && inputs[0].type !== 'float64') ||
      (inputs[1].type !== 'float32' && inputs[1].type !== 'float64')) {
    throw new Error('inputs should be float type');
  }

  if (inputs[0].type !== inputs[1].type) {
    throw new Error('inputs types should match');
  }
};
