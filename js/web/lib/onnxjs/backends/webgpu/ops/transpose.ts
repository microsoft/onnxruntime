// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../../../attribute-with-cache-key';
import {Graph} from '../../../graph';
import {OperatorAsyncImplementation, OperatorInitialization} from '../../../operators';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGpuInferenceHandler} from '../inference-handler';
import {GpuDataType, ProgramInfo} from '../types';

import {createIndicesHelper, generateDispatchGroup, WORKGROUP_SIZE} from './common';
import {declareDataSize, declareWorkgroupSize, mainBegin} from './shader-util';

export interface TransposeAttributes extends AttributeWithCacheKey {
  readonly perm: number[];
}

const transposeProgramMetadata = {
  name: 'Transpose',
  inputTypes: [GpuDataType.default]
};

export const transpose: OperatorAsyncImplementation<TransposeAttributes> = async(
    inferenceHandler: WebGpuInferenceHandler, inputs: Tensor[], attributes: TransposeAttributes): Promise<Tensor[]> => {
  validateInputs(inputs);
  return inferenceHandler.run(
      {
        ...transposeProgramMetadata,
        cacheHint: attributes.cacheKey,
        get: () => createTransposeProgramInfo(inferenceHandler, inputs[0], attributes.perm)
      },
      inputs);
};

export const parseTransposeAttributes: OperatorInitialization<TransposeAttributes> =
    (node: Graph.Node): TransposeAttributes => createAttributeWithCacheKey({perm: node.attributes.getInts('perm', [])});

const createTransposeProgramInfo =
    (_inferenceHandler: WebGpuInferenceHandler, input: Tensor, perm: number[]): ProgramInfo => {
      const dataType = 'f32';  // TODO: support other data type
      const inputShape = input.dims;
      perm = getAdjustedPerm(inputShape, perm);
      const outputShape = getOutputShape(inputShape, perm);
      const rank = inputShape.length;
      const outputSize = ShapeUtil.size(outputShape);
      // A dims=[${inputs[0].dims.toString()}]
      // out Dims=[${unpackedOutputShape.toString()}]
      // based on perm=[${perm.toString()}]

      const outputIndicesHelper = createIndicesHelper('output', outputShape);
      const inputIndicesHelper = createIndicesHelper('a', inputShape);

      const dispatchGroup = generateDispatchGroup(Math.ceil(outputSize / WORKGROUP_SIZE));
      const shaderSource = `
  ${declareWorkgroupSize()}
  ${declareDataSize(outputSize)}

  @group(0) @binding(0) var<storage, read> a : array<${dataType}>;
  @group(0) @binding(1) var<storage, read_write> output : array<${dataType}>;

  ${permFunctionBody(perm, rank)}
  ${outputIndicesHelper.o2iImpl}
  ${inputIndicesHelper.i2oImpl}

  ${mainBegin(dispatchGroup)}

    ${outputIndicesHelper.indicesVariableDeclaration('indices')}
    ${outputIndicesHelper.o2iCall('global_index', 'indices')}
    ${inputIndicesHelper.indicesVariableDeclaration('aIndices')}
    perm(&aIndices, &indices);

    output[global_index] = a[${inputIndicesHelper.i2oExpression('aIndices')}];
  }`;
      return {
        ...transposeProgramMetadata,
        outputs: [{dims: outputShape, type: input.type, gpuDataType: GpuDataType.default}],
        shaderSource,
        dispatchGroup
      };
    };

const getAdjustedPerm = (inputShape: readonly number[], perm: number[]): number[] => {
  if (perm && perm.length !== inputShape.length) {
    perm = [...(inputShape.keys())].reverse();
  }
  return perm;
};

const getOutputShape = (inputShape: readonly number[], perm: number[]): readonly number[] => {
  perm = getAdjustedPerm(inputShape, perm);
  return ShapeUtil.sortBasedOnPerm(inputShape, perm);
};

const permFunctionBody = (perm: number[], rank: number): string => {
  const reverseFunc = [];
  reverseFunc.push(`fn perm(a: ptr<function, array<u32, ${rank}>>, i: ptr<function, array<u32, ${rank}>>) {`);
  for (let i = 0; i < rank; ++i) {
    reverseFunc.push(`\t(*a)[${perm[i]}]=(*i)[${i}];`);
  }
  reverseFunc.push('\t}');
  return reverseFunc.join('\n');
};

const validateInputs = (inputs: Tensor[]): void => {
  if (!inputs || inputs.length !== 1) {
    throw new Error('Transpose requires 1 input.');
  }

  if (inputs[0].type !== 'float32' && inputs[0].type !== 'float64') {
    throw new Error('input should be float tensor');
  }
};
