// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Graph} from '../../../graph';
import {OperatorImplementation, OperatorInitialization} from '../../../operators';
import {Tensor} from '../../../tensor';
import {ProtoUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';

export const cast: OperatorImplementation<Tensor.DataType> =
    (inferenceHandler: WebGLInferenceHandler, inputs: Tensor[], to: Tensor.DataType): Tensor[] => {
      validateInputs(inputs);
      return [castTensor(inputs[0], to)];
    };

export const parseCastAttributes: OperatorInitialization<Tensor.DataType> = (node: Graph.Node): Tensor.DataType =>
    ProtoUtil.tensorDataTypeFromProto(node.attributes.getInt('to'));

const validateInputs = (inputs: Tensor[]): void => {
  if (!inputs || inputs.length !== 1) {
    throw new Error('Cast requires 1 input.');
  }

  if (inputs[0].type === 'string') {
    throw new Error('Invalid input type.');
  }
};

const castTensor = (input: Tensor, to: Tensor.DataType): Tensor => {
  const output = new Tensor([...input.dims], to);
  const inputData = input.data;
  const outputData = output.data;

  for (let i = 0; i < outputData.length; ++i) {
    outputData[i] = inputData[i];
  }

  return output;
};