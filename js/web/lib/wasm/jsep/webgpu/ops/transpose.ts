// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo} from '../types';

import {createTensorShapeVariables, IndicesHelper, inputVariable, outputVariable, ShaderHelper} from './common';

export interface TransposeAttributes extends AttributeWithCacheKey {
  readonly perm: number[];
}

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length !== 1) {
    throw new Error('Transpose requires 1 input.');
  }
};

const getAdjustedPerm = (inputRank: number, perm: number[]): number[] =>
    (perm && perm.length !== inputRank) ? [...(new Array(inputRank).keys())].reverse() : perm;

const getOutputShape = (inputShape: readonly number[], perm: number[]): readonly number[] =>
    ShapeUtil.sortBasedOnPerm(inputShape, getAdjustedPerm(inputShape.length, perm));

const permFunctionBody = (perm: number[], rank: number, input: IndicesHelper, output: IndicesHelper): string => {
  const reverseFunc = [];
  reverseFunc.push(`fn perm(i: ${output.type.indices}) -> ${input.type.indices} {
    var a: ${input.type.indices};`);
  for (let i = 0; i < rank; ++i) {
    reverseFunc.push(input.indicesSet('a', perm[i], `i[${i}]`));
  }
  reverseFunc.push('return a;}');
  return reverseFunc.join('\n');
};

export const createTransposeProgramInfo = (inputTensor: TensorView, permAttr: number[]): ProgramInfo => {
  const inputDataType = inputTensor.dataType;
  const inputRank = inputTensor.dims.length;
  const perm = getAdjustedPerm(inputRank, permAttr);
  const outputShape = getOutputShape(inputTensor.dims, perm);
  const output = outputVariable('output', inputDataType, outputShape.length);
  const input = inputVariable('a', inputDataType, inputRank);

  const getShaderSource = (shaderHelper: ShaderHelper) => `
  ${shaderHelper.registerUniform('output_size', 'u32').declareVariables(input, output)}

  ${permFunctionBody(perm, inputRank, input, output)}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.output_size')}

    let indices = ${output.offsetToIndices('global_idx')};
    let aIndices = perm(indices);

    ${output.setByOffset('global_idx', input.getByIndices('aIndices'))}
  }`;
  return {
    name: 'Transpose',
    shaderCache: {hint: `${permAttr}`, inputDependencies: ['rank']},
    getRunData: (inputs) => {
      const outputSize = ShapeUtil.size(outputShape);
      return {
        outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
        dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)},
        programUniforms:
            [{type: DataType.uint32, data: outputSize}, ...createTensorShapeVariables(inputs[0].dims, outputShape)],
      };
    },
    getShaderSource,
  };
};

export const transpose = (context: ComputeContext, attributes: TransposeAttributes): void => {
  validateInputs(context.inputs);
  context.compute(createTransposeProgramInfo(context.inputs[0], attributes.perm));
};

export const parseTransposeAttributes = (attributes: Record<string, unknown>): TransposeAttributes =>
    createAttributeWithCacheKey({perm: attributes.perm as number[]});
