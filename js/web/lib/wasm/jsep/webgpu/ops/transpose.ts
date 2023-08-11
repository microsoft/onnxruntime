// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo} from '../types';

import {IndicesHelper, inputVariable, outputVariable, ShaderHelper} from './common';

export interface TransposeAttributes extends AttributeWithCacheKey {
  readonly perm: number[];
}

export const transposeProgramMetadata = {
  name: 'Transpose',
  inputTypes: [GpuDataType.default]
};

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length !== 1) {
    throw new Error('Transpose requires 1 input.');
  }

  if (inputs[0].dataType !== DataType.float && inputs[0].dataType !== DataType.int32 &&
      inputs[0].dataType !== DataType.uint32) {
    throw new Error('Transpose only support float, int32, and uint32 data types');
  }
};

const getAdjustedPerm = (inputShape: readonly number[], perm: number[]): number[] =>
    (perm && perm.length !== inputShape.length) ? [...(inputShape.keys())].reverse() : perm;

const getOutputShape = (inputShape: readonly number[], perm: number[]): readonly number[] =>
    ShapeUtil.sortBasedOnPerm(inputShape, getAdjustedPerm(inputShape, perm));

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
  const dataType = inputTensor.dataType;
  const inputShape = inputTensor.dims;
  const perm = getAdjustedPerm(inputShape, permAttr);
  const outputShape = getOutputShape(inputShape, perm);
  const rank = inputShape.length;
  const outputSize = ShapeUtil.size(outputShape);
  // A dims=[${inputs[0].dims.toString()}]
  // out Dims=[${unpackedOutputShape.toString()}]
  // based on perm=[${perm.toString()}]

  const output = outputVariable('output', dataType, outputShape);
  const input = inputVariable('a', dataType, inputShape);

  const getShaderSource = (shaderHelper: ShaderHelper) => `
  ${shaderHelper.declareVariables(input, output)}

  ${permFunctionBody(perm, rank, input, output)}
  ${output.impl('offsetToIndices')}
  ${input.impl('indicesToOffset', 'get')}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

    let indices = ${output.offsetToIndices('global_idx')};
    let aIndices = perm(indices);

    ${output.setByOffset('global_idx', input.getByIndices('aIndices'))}
  }`;
  return {
    ...transposeProgramMetadata,
    outputs: [{dims: outputShape, dataType: inputTensor.dataType, gpuDataType: GpuDataType.default}],
    getShaderSource,
    dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
  };
};

export const transpose = (context: ComputeContext, attributes: TransposeAttributes): void => {
  validateInputs(context.inputs);
  context.compute({
    ...transposeProgramMetadata,
    cacheHint: attributes.cacheKey,
    get: () => createTransposeProgramInfo(context.inputs[0], attributes.perm)
  });
};

export const parseTransposeAttributes = (attributes: Record<string, unknown>): TransposeAttributes =>
    createAttributeWithCacheKey({perm: attributes.perm as number[]});
