// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo} from '../types';

import {createIndicesHelper, ShaderHelper} from './common';

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

  if (inputs[0].dataType !== DataType.float) {
    throw new Error('input should be float tensor');
  }
};

const getAdjustedPerm = (inputShape: readonly number[], perm: number[]): number[] =>
    (perm && perm.length !== inputShape.length) ? [...(inputShape.keys())].reverse() : perm;

const getOutputShape = (inputShape: readonly number[], perm: number[]): readonly number[] =>
    ShapeUtil.sortBasedOnPerm(inputShape, getAdjustedPerm(inputShape, perm));

const permFunctionBody = (perm: number[], rank: number): string => {
  const reverseFunc = [];
  reverseFunc.push(`fn perm(a: ptr<function, array<u32, ${rank}>>, i: ptr<function, array<u32, ${rank}>>) {`);
  for (let i = 0; i < rank; ++i) {
    reverseFunc.push(`\t(*a)[${perm[i]}]=(*i)[${i}];`);
  }
  reverseFunc.push('\t}');
  return reverseFunc.join('\n');
};

export const createTransposeProgramInfo = (input: TensorView, permAttr: number[]): ProgramInfo => {
  const dataType = 'f32';  // TODO: support other data type
  const inputShape = input.dims;
  const perm = getAdjustedPerm(inputShape, permAttr);
  const outputShape = getOutputShape(inputShape, perm);
  const rank = inputShape.length;
  const outputSize = ShapeUtil.size(outputShape);
  // A dims=[${inputs[0].dims.toString()}]
  // out Dims=[${unpackedOutputShape.toString()}]
  // based on perm=[${perm.toString()}]

  const outputIndicesHelper = createIndicesHelper('output', outputShape);
  const inputIndicesHelper = createIndicesHelper('a', inputShape);

  const getShaderSource = (shaderHelper: ShaderHelper) => `
  @group(0) @binding(0) var<storage, read> a : array<${dataType}>;
  @group(0) @binding(1) var<storage, read_write> output : array<${dataType}>;

  ${permFunctionBody(perm, rank)}
  ${outputIndicesHelper.o2iImpl}
  ${inputIndicesHelper.i2oImpl}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

    ${outputIndicesHelper.indicesVariableDeclaration('indices')}
    ${outputIndicesHelper.o2iCall('global_idx', 'indices')}
    ${inputIndicesHelper.indicesVariableDeclaration('aIndices')}
    perm(&aIndices, &indices);

    output[global_idx] = a[${inputIndicesHelper.i2oExpression('aIndices')}];
  }`;
  return {
    ...transposeProgramMetadata,
    outputs: [{dims: outputShape, dataType: input.dataType, gpuDataType: GpuDataType.default}],
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
