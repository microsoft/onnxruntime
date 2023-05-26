// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo} from '../types';
import {createIndicesHelper, ShaderHelper} from './common';
import {ShapeUtil} from '../../util';


const validateInputs = (inputs: readonly TensorView[]): void => {
    if (!inputs || inputs.length !== 1) {
        throw new Error('Pool ops requires 1 input.');
    }
    if (inputs[0].dataType !== DataType.float) {
        throw new Error('Invalid input type.');
    }
};

export interface SoftmaxAttributes extends AttributeWithCacheKey {
    readonly axis: number;
}

export const softmaxProgramMetadata = {
    name: 'Softmax',
    inputTypes: [GpuDataType.default]
};

const getOutputShape = (inputShape: readonly number[], axis: number): readonly number[] =>
    inputShape.slice(0, axis).concat(inputShape.slice(axis + 1));
export const createSoftmaxProgramInfo = (input: TensorView, axis: number): ProgramInfo => {
    const dataType = 'f32';  // TODO: support other data type
    const inputShape = input.dims;
    const outputShape = getOutputShape(inputShape, axis);
    const outputIndicesHelper = createIndicesHelper('output', outputShape);
    const inputIndicesHelper = createIndicesHelper('a', inputShape);
    const outputSize = ShapeUtil.size(outputShape);
    const getShaderSource = (shaderHelper: ShaderHelper) => `
  @group(0) @binding(0) var<storage, read> a : array<${dataType}>;
  @group(0) @binding(1) var<storage, read_write> output : array<${dataType}>;

  ${axis}
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
        ...softmaxProgramMetadata,
        outputs: [{dims: outputShape, dataType: input.dataType, gpuDataType: GpuDataType.default}],
        getShaderSource,
        dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
    };

};

export const softmax = (context: ComputeContext, attributes: SoftmaxAttributes): void => {
    validateInputs(context.inputs);
    context.compute(
        {
            ...softmaxProgramMetadata,
            cacheHint: attributes.cacheKey,
            get: () => createSoftmaxProgramInfo(context.inputs[0], attributes.axis)
        });
};

export const parseSoftmaxAttributes = (attributes: Record<string, unknown>): SoftmaxAttributes =>
    createAttributeWithCacheKey({axis: attributes.axis as number});
