// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Graph} from '../../../graph';
import {NUMBER_TYPES, OperatorImplementation, OperatorInitialization} from '../../../operators';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, TextureType} from '../types';

export const gather: OperatorImplementation<number> =
    (inferenceHandler: WebGLInferenceHandler, inputs: Tensor[], axis: number): Tensor[] => {
      validateInputs(inputs, axis);
      const output = inferenceHandler.run(createGatherProgramInfo(inferenceHandler, inputs, axis), inputs);
      return [output];
    };

export const parseGatherAttributes: OperatorInitialization<number> = (node: Graph.Node): number =>
    node.attributes.getInt('axis', 0);

const createGatherProgramInfo = (handler: WebGLInferenceHandler, inputs: Tensor[], axis: number): ProgramInfo => {
  const inputShape = inputs[0].dims.slice();
  const indexDataShape = inputs[1].dims.slice();
  const outputShape = new Array(inputShape.length + indexDataShape.length - 1);

  axis = ShapeUtil.normalizeAxis(axis, inputShape.length);
  const indexCopyOps: string[] = [];
  for (let i = 0; i < outputShape.length; i++) {
    // outputShape is divided into three parts: A, B, C
    // |0        axis|  axis + indexDataShape.length |          end|
    // |     A       |             B                 |      C      |
    //
    // inputIdx: [A, inputs[1][B], C]
    if (i < axis) {  // A
      outputShape[i] = inputShape[i];
      indexCopyOps.push(`inputIdx[${i}] = outputIdx[${i}];`);
    } else {
      if (i < axis + indexDataShape.length) {  // B
        outputShape[i] = indexDataShape[i - axis];
        indexCopyOps.push(`indexDataIdx[${i - axis}] = outputIdx[${i}];`);
      } else {                                                       // C
        outputShape[i] = inputShape[i - indexDataShape.length + 1];  // skip 1 for axis
        indexCopyOps.push(`inputIdx[${i - indexDataShape.length + 1}] = outputIdx[${i}];`);
      }
    }
  }

  const orank = outputShape.length || 1;
  const irank = inputShape.length;
  const iDrank = indexDataShape.length || 1;
  const shaderSource = `
      float process(int outputIdx[${orank}]) {
        int inputIdx[${irank}];
        int indexDataIdx[${iDrank}];
        indexDataIdx[0] = 0;
        ${indexCopyOps.join('\n        ')}
        int idx = int(_B(indexDataIdx));
        inputIdx[${axis}] = idx < 0 ? idx + ${inputShape[axis]} : idx;
        return _A(inputIdx);
      }`;
  return {
    name: 'Gather',
    inputNames: ['A', 'B'],
    inputTypes: [TextureType.unpacked, TextureType.unpacked],
    output: {dims: outputShape, type: inputs[0].type, textureType: TextureType.unpacked},
    shaderSource
  };
};

const validateInputs = (inputs: Tensor[], axis: number): void => {
  if (!inputs || inputs.length !== 2) {
    throw new Error('Gather requires 2 inputs.');
  }
  const tensorRank = inputs[0].dims.length;
  if (tensorRank < 1) {
    throw new Error('Invalid input shape.');
  }
  if (axis < -tensorRank || axis > tensorRank - 1) {
    throw new Error('Invalid axis.');
  }
  if (NUMBER_TYPES.indexOf(inputs[0].type) === -1) {
    throw new Error('Invaid input type.');
  }
  if (inputs[1].type !== 'int32' && inputs[1].type !== 'int16') {
    throw new Error('Invaid input type.');
  }
};